"""Full-epoch, resumable Section 8–9 CFM training.

The two-time pilot convention is embedding(t), embedding(0): CFM has r=t.
No VAE, token masking, classifier-free guidance, curriculum, AMP or EMA.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def embedding(t, width=64):
    frequencies = torch.exp(-math.log(10000) * torch.arange(
        width // 2, device=t.device, dtype=torch.float32) / (width // 2))
    angles = t[:, None].float() * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width),
                                 nn.GELU(approximate="tanh"), nn.Linear(4 * width, width))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, c):
        s1, a1, g1, s2, a2, g2 = self.mod(c).chunk(6, -1)
        h = self.norm1(x) * (1 + a1[:, None]) + s1[:, None]
        batch, tokens, width = h.shape
        q, k, v = self.qkv(h).reshape(batch, tokens, 3, self.heads,
                                    width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # FP32 SDPA preserves the pilot's softmax(QK^T/sqrt(d)) V mathematics.
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h.transpose(1, 2).reshape(batch, tokens, width)
        x = x + g1[:, None] * self.proj(h)
        return x + g2[:, None] * self.mlp(self.norm2(x) * (1 + a2[:, None]) + s2[:, None])


class DiT(nn.Module):
    def __init__(self, patch=4, width=128, depth=4, heads=4):
        super().__init__()
        if patch not in (2, 4) or width <= 0 or depth <= 0 or heads <= 0 or width % heads:
            raise ValueError("patch must be 2/4 and positive width divisible by positive heads")
        self.patch_size = patch
        self.grid_size = 64 // patch
        self.num_tokens = self.grid_size ** 2
        self.patch = nn.Linear(2 * patch * patch, width)
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, width) * .02)
        self.time = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.cond = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 2 * patch * patch)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def patchify(self, z):
        if z.ndim != 4 or tuple(z.shape[1:]) != (2, 64, 64):
            raise ValueError("Expected [N,2,64,64]")
        b, p, g = len(z), self.patch_size, self.grid_size
        return z.reshape(b, 2, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(b, g*g, 2*p*p)

    def unpatchify(self, x):
        b, p, g = len(x), self.patch_size, self.grid_size
        return x.reshape(b, g, g, 2, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, 2, 64, 64)

    def forward(self, z, t, odt):
        if tuple(t.shape) != (len(z),) or tuple(odt.shape) != (len(z), 5):
            raise ValueError("Expected per-query t[N] and odt[N,5]")
        x = self.patch(self.patchify(z)) + self.pos
        c = self.time(torch.cat((embedding(t), embedding(torch.zeros_like(t))), -1)) + self.cond(odt)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.out(self.norm(x)))


@torch.no_grad()
def sample(model, odt, noise, nfe=1):
    if not isinstance(nfe, int) or nfe < 1:
        raise ValueError("nfe must be a positive integer")
    z = noise.clone()
    for i in range(nfe):
        t = torch.full((len(odt),), 1 - i / nfe, device=z.device, dtype=z.dtype)
        z = z - model(z, t, odt) / nfe
    return z


def cfm_loss(model, image, odt, noise, times, fg_weight):
    if not math.isfinite(float(fg_weight)) or fg_weight <= 0:
        raise ValueError("fg_weight must be finite and positive")
    t = times[:, None, None, None]
    z = (1 - t) * image + t * noise
    error = (model(z, times, odt) - (noise - image)).square()
    weights = torch.where(image[:, :1] > 0, float(fg_weight), 1.)
    # Both channel errors share occupancy weights. Normalize each query first.
    return ((error * weights).flatten(1).sum(1) /
            (2 * weights.flatten(1).sum(1))).mean()


def lr_at_update(update, peak_lr, total_updates, schedule="constant", warmup=100):
    """One-based update rate, with the horizon fixed before the first update."""
    if update < 1 or total_updates < 1 or update > total_updates or warmup < 1 or peak_lr <= 0:
        raise ValueError("Invalid update, horizon, warmup or peak learning rate")
    if schedule not in ("constant", "cosine"):
        raise ValueError("Unsupported schedule")
    if update <= warmup:
        return peak_lr * update / warmup
    if schedule == "constant":
        return peak_lr
    fraction = (update - warmup) / (total_updates - warmup)
    return peak_lr * (.1 + .9 * (1 + math.cos(math.pi * fraction)) / 2)


def draw_noise_times(image, noise_generator, time_generator):
    noise = torch.randn(image.shape, device=image.device, dtype=torch.float32, generator=noise_generator)
    n = len(image)
    u1 = torch.rand(n, device=image.device, generator=time_generator)
    u2 = torch.rand(n, device=image.device, generator=time_generator)
    choose_max = torch.rand(n, device=image.device, generator=time_generator) < .5
    return noise, torch.where(choose_max, torch.maximum(u1, u2), u1)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed on resume")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def _evaluation_context(model):
    state, training = _rng_state(), model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        _restore_rng(state)
        model.train(training)


def _atomic_write(path, value, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as handle:
            if binary:
                torch.save(value, handle)
            else:
                json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


