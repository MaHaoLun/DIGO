"""Predict an interval mask and return the exact regional route."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/g0'))
sys.path.insert(0, str(ROOT / 'ges'))
from evaluator64 import fixed_query_noise
from model64 import DiT, sample
from native import Native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('graph', 'library', 'checkpoint', 'conditions', 'city', 'snapshot', 'role'):
        parser.add_argument('--' + name, required=True)
    for name in ('source', 'destination', 'lo', 'hi'):
        parser.add_argument('--' + name, required=True, type=int)
    parser.add_argument('--threshold', required=True, type=float)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiT().to(device)
    weights = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(weights.get('model', weights))
    model.eval()
    with np.load(args.conditions, allow_pickle=False) as data:
        odt, ids = data['odt'], data['query_ids']
        support, ends = data['support'], data['endpoints']
    if len(ids) == 0 or odt.shape != (len(ids), 5) or support.size != 4096 or ends.shape != (2,):
        parser.error('expected N conditions for one OD, 4096 support cells and two endpoint indices')
    if np.any(ends < 0) or np.any(ends >= 4096):
        parser.error('endpoint indices must be in [0,4095]')
    mask = np.zeros(4096, dtype=bool)
    with torch.no_grad():
        for i in range(0, len(ids), 2):
            noise = fixed_query_noise(ids[i:i+2], eval_seed=20260912, city=args.city,
                                      snapshot_id=args.snapshot, role=args.role)
            raw = sample(model, torch.as_tensor(odt[i:i+2], dtype=torch.float32, device=device),
                         torch.as_tensor(noise, device=device))
            scores = ((raw[:, 0] + 1) / 2).clamp(0, 1).flatten(1).cpu().numpy()
            mask |= (scores.astype(np.float64) >= args.threshold).any(axis=0)
    mask &= support.reshape(-1).astype(bool)
    mask[ends] = True
    solver = Native(args.graph, args.library)
    try:
        print(json.dumps(solver.query(args.source, args.destination, args.lo, args.hi, 4, mask.tolist())))
    finally:
        solver.close()


if __name__ == '__main__':
    main()
