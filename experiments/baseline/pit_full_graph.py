"""Candidate: capture the entire unchanged source sampler, including its RNG.

PyTorch 2.6 CUDAGraph.register_generator_state is present on the execution host.
The registered generator remains private to this graph and is seeded before
each replay. No denoiser call, reverse step, or random draw is removed.
This module is NOT enabled in formal inference without separate gates.
"""
import time
import torch
import models


class FullSampler:
    @torch.no_grad()
    def __init__(self, model, process, condition):
        assert not model.training and condition.device.type == 'cuda'
        self.condition = condition.clone()
        self.generator = torch.Generator(device=condition.device).manual_seed(42)
        self.batch = len(condition)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.register_generator_state(self.generator)
        # Source comparisons precede construction in every checker/worker.
        # Warm the denoiser on the same separate capture stream as GraphModel.
        stream = torch.cuda.Stream(device=condition.device)
        stream.wait_stream(torch.cuda.current_stream(condition.device))
        with torch.cuda.stream(stream):
            x = torch.zeros((self.batch, 3, 64, 64), device=condition.device)
            t = torch.full((self.batch,), 500, device=condition.device, dtype=torch.long)
            for _ in range(3):
                model(x, t, self.condition)
        torch.cuda.current_stream(condition.device).wait_stream(stream)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.output = models.pit_sample(model, process, self.condition, self.generator)
        torch.cuda.synchronize()
        self.capture_seconds = time.monotonic() - started

    @torch.no_grad()
    def sample(self, condition, seed):
        assert condition.shape == self.condition.shape
        self.condition.copy_(condition)
        self.generator.manual_seed(seed)
        self.graph.replay()
        # Preserve each output while the same graph is reused for another OD.
        return self.output.clone()
