"""Independent scalar legacy-detector reconstruction and full role checks."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np


def canonical(gids, n):
    s, d = gids//n, gids % n
    return np.minimum(s, d)*n+np.maximum(s, d)


def changed(a, b):
    absolute = sum(abs(int(y)-int(x)) for x, y in zip(a, b))
    denominator = max(sum(map(int, a)), 3)
    return absolute >= 15 and 10*absolute >= denominator


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run
    config = json.loads((run/'protocol.json').read_text()); parent = Path(config['parent_e1_run'])
    sys.path.insert(0, str(parent/'code'))
    from p0_inventory import atomic
    from p0_plan import role_codes
    assert not changed([100]*3, [105]*3)  # Absolute passes, relative fails.
    assert changed([50]*3, [55]*3)  # Both exact boundaries pass.
    assert not changed([5]*3, [9]*3)  # Relative passes, absolute fails.
    assert not changed([51]*3, [56]*3)
    assert changed([49]*3, [54]*3)
    assert not changed([1]*3, [1]*3)
    assert [(n+4)//5 for n in (0, 1, 4, 5, 6)] == [0, 1, 1, 1, 2]
    results = []
    for city in ('XA', 'BJ'):
        folder = run/'data'/city
        report = json.loads((folder/'static_preflight.json').read_text())
        assert report['status'] == 'STATIC_PREFLIGHT_PASSED_TARGETS_PENDING' and report['protocol'] == config
        coords = np.load(folder/'coordinates.npy'); n = len(coords); cells = np.load(folder/'cell64.npy')
        old = np.load(folder/'G0_graph.npz'); new = np.load(folder/'G1_graph.npz')
        u, v = new['u'], new['v']; a, b = old['knots'], new['knots']
        assert np.array_equal(old['u'], u) and np.array_equal(old['v'], v)
        scores = [0.]*4096; significant = []
        for index in np.flatnonzero(u < v):
            before = [int(a[index, k]) for k in (48, 49, 50)]
            after = [int(b[index, k]) for k in (48, 49, 50)]
            if not changed(before, after): continue
            absolute = sum(abs(y-x) for x, y in zip(before, after))
            relative = absolute/max(sum(before), 3)
            first, second = int(cells[u[index]]), int(cells[v[index]])
            scores[first] += relative
            if first != second: scores[second] += relative
            significant.append(int(u[index])*n+int(v[index]))
        order = sorted((i for i in range(4096) if scores[i] > 0), key=lambda i:(-scores[i], i))
        expected_hot = np.asarray(order[:(len(order)+4)//5], np.int64)
        saved = np.load(folder/'detection.npz')
        assert np.array_equal(saved['cell_scores'], np.asarray(scores))
        assert np.array_equal(saved['hotspots'], expected_hot)
        assert np.array_equal(saved['shifted_pair_keys'], significant)
        keys = np.minimum(u, v).astype(np.int64)*n+np.maximum(u, v)
        assert np.array_equal(saved['shifted_arcs'], np.isin(keys, significant))
        g0_groups = np.load(parent/'data'/city/'groups.npy')
        g0_ids = g0_groups[:, 0]*n+g0_groups[:, 1]
        base_ids = np.load(parent/'data'/city/'base_test_group_ids.npy')
        g0_test = canonical(g0_ids[base_ids], n)
        g0_all = canonical(g0_ids, n)
        c32 = np.load(folder/'cell32.npy'); cases = []
        for name, role in (('sdc_Afit', 0), ('Aselect', 1), ('Acal', 2), ('mixed_test', 4)):
            ordinary = np.load(folder/(name+'_ordinary_gids.npy'))
            targets = np.load(folder/(name+'_target_order_gids.npy'))
            spec = report['roles'][name]
            assert len(ordinary) == spec['ordinary_groups']
            assert len(targets) == spec['target_candidates']
            assert len(ordinary)+len(targets) == spec['legal_groups']
            assert len(np.intersect1d(ordinary, targets)) == 0
            gids = np.concatenate((ordinary, targets))
            assert len(np.unique(gids)) == len(gids)
            src, dst = gids//n, gids % n
            assert np.all(role_codes(src, dst, n) == role)
            distance = np.abs(c32[src] % 32-c32[dst] % 32)+np.abs(c32[src]//32-c32[dst]//32)
            assert np.all((distance >= 49) & (distance <= 56))
            priority = saved['cell_scores'][cells[targets//n]]+saved['cell_scores'][cells[targets % n]]
            assert np.all(priority[:-1] >= priority[1:])
            if role == 4:
                assert len(np.intersect1d(canonical(gids, n), g0_all)) == 0
            else:
                assert len(np.intersect1d(canonical(gids, n), g0_test)) == 0
            cases.append(dict(role=name, code=role, candidate_groups=len(gids),
                ordinary_groups=len(ordinary), target_eligible_capacity='PENDING_GT'))
        random = np.load(folder/'random_Afit_gids.npy'); src, dst = random//n, random % n
        assert len(random)*2 == config['budgets'][city]['Afit_points_per_FT']
        assert len(np.unique(random)) == len(random) and np.all(role_codes(src, dst, n) == 0)
        assert len(np.intersect1d(canonical(random, n), g0_test)) == 0
        d = np.abs(c32[src] % 32-c32[dst] % 32)+np.abs(c32[src]//32-c32[dst]//32)
        assert np.all((d >= 49) & (d <= 56))
        results.append(dict(city=city, status='PASSED', scalar_detector_bitwise_matches=True,
            hotspot_cells=len(expected_hot), significant_pairs=len(significant), full_role_candidates_checked=cases,
            random_fit_groups=len(random), G1_test_candidate_domain_absent_from_all_G0_labels=True,
            G0_test_absent_from_all_G1_adaptation_candidates=True))
        old.close(); new.close(); saved.close()
    result = dict(status='PASSED', results=results, recorded_unix=time.time(),
        target_qualification_capacity='PENDING_GT', final_shared_holdout_frozen=False,
        formal_point_labels_computed=0, formal_training_updates=0, no_hashes=True)
    atomic(run/'audits/static_preflight_validation.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
