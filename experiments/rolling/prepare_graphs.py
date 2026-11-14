"""Release precisely one raw G1 knot per slot, then apply periodic FIFO closure."""
import argparse
import fcntl
import json
import os
import time
import numpy as np
from common import ROOT, PARENT, OLD, atomic, stamp


def fifo_closure(raw):
    fixed = np.ascontiguousarray(raw.T)
    for _ in range(2):
        for j in range(96):
            np.maximum(fixed[j], fixed[(j - 1) % 96] - 900, out=fixed[j])
    fixed = fixed.T.copy()
    assert np.all(fixed >= raw)
    assert np.all(fixed - np.roll(fixed, -1, axis=1) <= 900)
    changed = np.flatnonzero(np.any(fixed != raw, axis=1))
    if len(changed):
        selected = raw[changed]
        exact = selected.copy()
        for lag in range(1, 96):
            np.maximum(exact, np.roll(selected, lag, axis=1) - 900 * lag, out=exact)
        assert np.array_equal(exact, fixed[changed])
    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['BJ', 'XA'], required=True)
    args = parser.parse_args()
    city = args.city
    folder = ROOT / 'graphs' / city
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        zero = PARENT / f'e1_v1/data/{city}/routing_graph.npz'
        one = PARENT / f'e2_v1/data/{city}/G1_graph.npz'
        g0, g1 = dict(np.load(zero)), dict(np.load(one))
        assert np.array_equal(g0['u'], g1['u']) and np.array_equal(g0['v'], g1['v'])
        assert np.array_equal(fifo_closure(g0['knots']), g0['knots'])
        assert np.array_equal(fifo_closure(g1['knots']), g1['knots'])
        reports = []
        started = time.monotonic()
        for slot in range(1, 97):
            path = folder / f'v{slot:03d}.npz'
            recordpath = folder / f'v{slot:03d}.json'
            if recordpath.exists():
                report = json.loads(recordpath.read_text())
                assert stamp(path) == report['graph']
                reports.append(report)
                continue
            raw = g0['knots'].copy()
            raw[:, :slot] = g1['knots'][:, :slot]
            fixed = fifo_closure(raw)
            report = dict(status='GRAPH_PREPARED_FOR_INDEPENDENT_REVIEW', city=city, slot=slot,
                released_raw_knot_index=slot - 1, G1_released_knots=slot,
                release_available_clock_seconds=(slot - 1) * 900,
                evaluate_departure_seconds=(slot - 1) * 900,
                next_release_clock_seconds=slot * 900,
                fifo_raised_knots=int(np.count_nonzero(fixed != raw)),
                maximum_fifo_raise=int((fixed - raw).max()), periodic_FIFO=True,
                exact_minimum_majorant_checked=True, source0=stamp(zero), source1=stamp(one),
                final_equals_G1=slot == 96, matches_legacy_hourly=slot % 4 == 0,
                solver_calls=0, model_calls=0, hash_verification=False)
            if slot % 4 == 0:
                prior = OLD / f'graphs/{city}/v{slot // 4:02d}.npz'
                with np.load(prior) as old:
                    assert np.array_equal(old['u'], g0['u']) and np.array_equal(old['v'], g0['v'])
                    assert np.array_equal(old['knots'], fixed)
                if not path.exists():
                    path.symlink_to(prior)
                report['exact_legacy_graph_reuse'] = stamp(prior)
            else:
                temp = path.with_suffix('.tmp.npz')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream, u=g0['u'], v=g0['v'], knots=fixed)
                temp.replace(path)
            if slot == 96:
                assert np.array_equal(fixed, g1['knots'])
            report['graph'] = stamp(path)
            atomic(recordpath, report)
            reports.append(report)
            atomic(folder / 'progress.json', dict(status='GENERATING', city=city,
                slots=len(reports), required_slots=96, pid=os.getpid(), updated_unix=time.time(),
                elapsed_seconds=time.monotonic() - started))
        atomic(folder / 'prepared.json', dict(status='ALL96_GRAPHS_READY_FOR_INDEPENDENT_REVIEW',
            city=city, slots=96, reports=reports, previous_hourly_matches=24,
            final_equals_G1=True, solver_calls=0, model_calls=0, hash_verification=False))
        print(json.dumps(dict(status='ALL96_GRAPHS_READY_FOR_INDEPENDENT_REVIEW', city=city)), flush=True)


if __name__ == '__main__':
    main()
