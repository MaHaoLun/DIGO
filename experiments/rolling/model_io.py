"""SSP prediction records; fixed noise, complete counts and no repeated forwards."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from common import PARENT, atomic, frozen_json, frozen_npz, stamp
sys.path.insert(0, str(PARENT / 'e1_v1/code'))
import model64 as mt
import evaluator64 as ev


def save_pt(path, value, sync=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        torch.save(value, stream)
        if sync:
            stream.flush()
            os.fsync(stream.fileno())
    tmp.replace(path)


def counts_summary(scores, data, support, threshold):
    gt = data['image'][:, 0].reshape(len(scores), 4096) > 0
    masks = (scores.astype(np.float64) >= threshold) & support
    masks[np.arange(len(masks))[:, None], data['endpoints']] = True
    counts = np.column_stack(((masks & gt).sum(1), (masks & ~gt).sum(1), (~masks & gt).sum(1)))
    tp, fp, fn = map(int, counts.sum(0))
    denominator = 2 * counts[:, 0] + counts[:, 1] + counts[:, 2]
    point_f1 = np.divide(2 * counts[:, 0], denominator, out=np.zeros(len(counts), float), where=denominator != 0)
    return dict(points=len(scores), tp=tp, fp=fp, fn=fn,
        precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn),
        f1=2 * tp / (2 * tp + fp + fn), macro_point_f1=float(point_f1.mean()),
        path_coverage=float(np.mean(counts[:, 2] == 0)), threshold=float(threshold)), counts, masks


def evaluate(model, data, out, city, slot, role, support, model_identity, threshold=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(data['query_ids'])
    assert n % 2 == 0
    identity = dict(city=city, slot=slot, role=role, model=model_identity,
        query_ids=data['query_ids'].tolist(), batch=2, nfe=1,
        noise_tag=f'{city}_rolling900_v{slot:03d}', noise_role=role, noise_seed=20260912)
    frozen_json(out / 'identity.json', identity)
    if (out / 'metrics.json').exists():
        result = json.loads((out / 'metrics.json').read_text())
        assert result['points'] == n
        if threshold is not None:
            assert result['threshold'] == threshold
        return result
    scores = np.empty((n, 4096), dtype=np.float32)
    original_training, rng = model.training, mt._rng_state()
    seconds = 0.
    model.eval()
    try:
        with torch.no_grad():
            for begin in range(0, n, 2):
                path = out / f'batch_{begin:04d}.npz'
                intent = path.with_suffix('.intent.json')
                qids = data['query_ids'][begin:begin + 2]
                if path.exists():
                    with np.load(path) as saved:
                        assert np.array_equal(saved['query_ids'], qids)
                        scores[begin:begin + 2] = saved['scores']
                        seconds += float(saved['pipeline_seconds'])
                    continue
                assert not intent.exists(), 'Unresolved forward intent: no automatic repeat'
                frozen_json(intent, dict(status='FORWARD_INTENT', batch_start=begin,
                    query_ids=qids.tolist(), model=model_identity, pid=os.getpid(),
                    started_unix=time.time()))
                noise = ev.fixed_query_noise(qids, eval_seed=20260912, city=city,
                    snapshot_id=identity['noise_tag'], role=role, shape=(2, 64, 64))
                torch.cuda.synchronize()
                started = time.monotonic()
                raw = mt.sample(model, torch.from_numpy(data['odt'][begin:begin + 2]).cuda(),
                    torch.from_numpy(noise).cuda(), nfe=1)
                values = np.clip((raw[:, 0].cpu().numpy() + np.float32(1)) / np.float32(2), 0, 1).reshape(2, 4096)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                assert np.isfinite(values).all()
                frozen_npz(path, query_ids=qids, scores=values, pipeline_seconds=elapsed)
                scores[begin:begin + 2] = values
                seconds += elapsed
    finally:
        model.train(original_training)
        mt._restore_rng(rng)
    sweep_seconds = 0.
    if threshold is None:
        started = time.monotonic()
        gt = data['image'][:, 0].reshape(n, 4096) > 0
        sweep = ev.exact_threshold_sweep(scores, gt, data['endpoints'], support)
        threshold = float(sweep['best']['threshold'])
        sweep_seconds = time.monotonic() - started
        atomic(out / 'calibration.json', {k: v for k, v in sweep.items() if k != 'coarse_curve'})
    result, counts, masks = counts_summary(scores, data, support, threshold)
    np.save(out / 'scores.npy', scores)
    np.save(out / 'point_counts.npy', counts)
    np.save(out / 'point_masks_packbits.npy', np.packbits(masks, axis=1))
    result = dict(status='SAVED_PREDICTIONS_PENDING_INDEPENDENT_RECOUNT', **result,
        city=city, slot=slot, clock_seconds=(slot - 1) * 900 if slot else 0,
        role=role, prediction_pipeline_seconds=seconds, threshold_sweep_seconds=sweep_seconds)
    atomic(out / 'metrics.json', result)
    return result
