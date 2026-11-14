"""E2 static capacity and native64 shift detector, before any GT qualification."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def save_array(path, value):
    with path.with_suffix('.tmp.npy').open('wb') as out:
        np.save(out, value)
    path.with_suffix('.tmp.npy').replace(path)


def graph_policy(snapshot, coords, graph):
    """Independent canonical-row/least-upward-majorant check, including G1."""
    original = np.loadtxt(snapshot, dtype=np.int64, skiprows=1)
    n = len(coords)
    u = np.searchsorted(coords[:, 0], original[:, 0])
    v = np.searchsorted(coords[:, 0], original[:, 1])
    assert np.array_equal(coords[u, 0], original[:, 0])
    assert np.array_equal(coords[v, 0], original[:, 1])
    assert np.all(u != v), 'Self loops need an explicit canonical detector rule'
    pick = u < v
    keys = (u.astype(np.int64)*n+v)[pick]
    order = np.argsort(keys)
    canonical = original[pick, 2:][order]
    actual_pick = graph.u < graph.v
    assert np.array_equal(graph.keys[actual_pick], keys[order])
    repaired = graph.knots[actual_pick]
    assert canonical.shape == repaired.shape and np.all(repaired >= canonical)
    changed = np.flatnonzero(np.any(repaired != canonical, axis=1))
    if len(changed):
        raw = canonical[changed]
        expected = raw.copy()
        for distance in range(1, 96):
            np.maximum(expected, np.roll(raw, distance, axis=1)-900*distance, out=expected)
        assert np.array_equal(repaired[changed], expected)
    reverse = np.searchsorted(graph.keys, graph.v.astype(np.int64)*n+graph.u)
    assert np.array_equal(graph.keys[reverse], graph.v.astype(np.int64)*n+graph.u)
    assert np.array_equal(graph.knots[reverse], graph.knots)
    assert np.all(graph.knots > 0)
    assert np.all(graph.knots-np.roll(graph.knots, -1, axis=1) <= 900)
    return dict(status='PASSED', directed_arcs=len(original), canonical_pairs=len(canonical),
                changed_pairs=len(changed), changed_knots=int(np.count_nonzero(repaired != canonical)),
                maximum_knot_raise=int((repaired-canonical).max()),
                independent_least_upward_majorant=True, reciprocal=True)


def detector(old, new, cells):
    assert np.array_equal(old.keys, new.keys)
    pick = new.u < new.v
    u, v = new.u[pick], new.v[pick]
    before = old.knots[pick][:, [48, 49, 50]]
    after = new.knots[pick][:, [48, 49, 50]]
    absolute_sum = np.abs(after-before).sum(1, dtype=np.int64)
    denominator = np.maximum(before.sum(1, dtype=np.int64), 3)
    significant = (absolute_sum >= 15) & (10*absolute_sum >= denominator)
    relative = absolute_sum[significant]/denominator[significant]
    a, b = cells[u[significant]], cells[v[significant]]
    # Preserve canonical-pair/first-endpoint/second-endpoint accumulation order.
    pair_cells = np.column_stack((a, b)).reshape(-1)
    pair_keep = np.column_stack((np.ones(len(a), bool), a != b)).reshape(-1)
    scores = np.bincount(pair_cells[pair_keep], weights=np.repeat(relative, 2)[pair_keep], minlength=4096)
    positive = np.flatnonzero(scores > 0)
    order = positive[np.lexsort((positive, -scores[positive]))]
    hotspots = order[:(len(order)+4)//5]
    shifted_keys = old.keys[pick][significant]
    canonical_keys = np.minimum(new.u, new.v).astype(np.int64)*new.n+np.maximum(new.u, new.v)
    shifted_arcs = np.isin(canonical_keys, shifted_keys)
    assert int(shifted_arcs.sum()) == 2*int(significant.sum())
    return dict(scores=scores, hotspots=hotspots, shifted_arcs=shifted_arcs,
                shifted_pair_keys=shifted_keys,
                summary=dict(status='PASSED', monitoring_times=[43200, 44100, 45000],
                    significant_pairs=int(significant.sum()), positive_cells=len(positive),
                    hotspot_cells=len(hotspots), monitoring_changed_pairs=int(np.count_nonzero(absolute_sum)),
                    effective_changed_anchor_pairs=int(np.count_nonzero(np.any(old.knots[pick] != new.knots[pick], axis=1))),
                    threshold='sum_abs>=15 AND 10*sum_abs>=max(sum_old,3)',
                    ranking='descending relative-sum score; ascending cell ID ties',
                    native_grid=64, no_GT_queries_computed=True))


def representatives(coords, box, c128):
    xy = coords[:, 1:]-[box[0], box[2]]
    result = []
    for cell in np.unique(c128):
        ix = np.flatnonzero(c128 == cell)
        cx, cy = int(cell % 128), int(cell // 128)
        dx = xy[ix, 0]*256-(2*cx+1)*(box[1]-box[0])
        dy = xy[ix, 1]*256-(2*cy+1)*(box[3]-box[2])
        result.append(ix[np.argmin(dx*dx+dy*dy)])
    return np.sort(np.asarray(result, np.int64))


def uniform_fit(n, cells32, components, exposed, wanted, rng, role_codes):
    result, used = [], set()
    while len(result) < wanted:
        src, dst = rng.integers(0, n, size=(2, 131072))
        distance = np.abs(cells32[src] % 32-cells32[dst] % 32)+np.abs(cells32[src]//32-cells32[dst]//32)
        keep = (distance >= 49) & (distance <= 56) & (components[src] == components[dst])
        src, dst = src[keep], dst[keep]
        keep = role_codes(src, dst, n) == 0
        for s, d in zip(src[keep], dst[keep]):
            s, d = int(s), int(d)
            gid = s*n+d
            if gid in used or (min(s, d), max(s, d)) in exposed:
                continue
            used.add(gid); result.append(gid)
            if len(result) == wanted:
                break
    return np.asarray(result, np.int64)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    args = p.parse_args(); run = args.run
    config = json.loads((run/'protocol.json').read_text())
    assert run == Path(config['run'])
    assert config['native_prediction_grid'] == 64 and config['sampling_subgrid'] == 128
    assert config['distance_grid'] == 32 and config['distance_range'] == [49, 56]
    assert config['departures'] == [43200, 44100] and config['monitoring_times'] == [43200, 44100, 45000]
    assert config['detector']['absolute_shift_seconds'] == 5 and config['detector']['relative_shift_threshold'] == .1
    assert config['detector']['hotspot_fraction'] == .2 and config['sampler']['targeted_fraction'] == .2
    parent = Path(config['parent_e1_run']); root = Path(config['remote_root'])
    assert json.loads((parent/'E1_ACCEPTANCE.json').read_text())['status'] == 'ACCEPTED'
    assert config['training_seeds'] == [42] and not config['hash_verification']
    sys.path.insert(0, str(parent/'code'))
    from p0_inventory import BBOX, atomic, cell_xy
    from p0_plan import candidates, role_codes, SPLIT_SEED
    from point_data import Native
    assert config['role_partition']['split_seed'] == SPLIT_SEED
    summaries = []
    for city, city_code in (('XA', 0), ('BJ', 1)):
        folder = run/'data'/city; folder.mkdir(parents=True, exist_ok=True)
        report_path = folder/'static_preflight.json'
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            assert prior['protocol'] == config and prior['status'] == 'STATIC_PREFLIGHT_PASSED_TARGETS_PENDING'
            summaries.append(prior); continue
        started = time.monotonic(); cpu_started = time.process_time()
        coordinates = np.load(parent/'data'/city/'coordinates.npy')
        n = len(coordinates); box = BBOX[city]
        stages = {stage:root/'project/digo_dataset_redesign_v1/data/staged'/(
            f'{city}_g0_offline_20260908' if stage == 'G0' else f'{city}_g1_online_20260911') for stage in ('G0', 'G1')}
        paths = [stages[s]/name for s in stages for name in ('snapshot.txt', 'coordinates.txt')]
        stamps = {str(q):[q.stat().st_size, q.stat().st_mtime_ns] for q in paths}
        assert np.array_equal(np.loadtxt(stages['G1']/'coordinates.txt', dtype=np.int64, skiprows=1), coordinates)
        old, new = Native(root, city, 'G0'), Native(root, city, 'G1')
        assert old.n == new.n == n and np.array_equal(old.keys, new.keys)
        with np.load(parent/'data'/city/'routing_graph.npz') as frozen:
            assert np.array_equal(frozen['u'], old.u) and np.array_equal(frozen['v'], old.v)
            assert np.array_equal(frozen['knots'], old.knots)
        graph_checks = {stage:graph_policy(stages[stage]/'snapshot.txt', coordinates, g)
                        for stage, g in (('G0', old), ('G1', new))}
        mappings = {}
        for size in (32, 64, 128):
            xy = cell_xy(coordinates[:, 1:], box, size)
            mappings[size] = xy[:, 1]*size+xy[:, 0]
        assert np.array_equal(mappings[64], np.load(parent/'data'/city/'cell64.npy'))
        change = detector(old, new, mappings[64])
        np.savez(folder/'G0_graph.npz', u=old.u, v=old.v, knots=old.knots)
        np.savez(folder/'G1_graph.npz', u=new.u, v=new.v, knots=new.knots)
        np.savez(folder/'detection.npz', cell_scores=change['scores'], hotspots=change['hotspots'],
                 shifted_arcs=change['shifted_arcs'], shifted_pair_keys=change['shifted_pair_keys'])
        atomic(folder/'detection.json', change['summary'])
        for name, value in (('coordinates', coordinates), ('cell32', mappings[32]),
                            ('cell64', mappings[64]), ('cell128', mappings[128])):
            save_array(folder/(name+'.npy'), value)
        graph = csr_matrix((np.ones(len(new.u), np.uint8), (new.u, new.v)), shape=(n, n))
        assert (graph != graph.T).nnz == 0
        _, components = connected_components(graph, directed=True, connection='strong')
        cap = json.loads((parent/'p0'/f'{city}_sampler_capacity.json').read_text())
        exposed = set()
        for entry in cap['exposure_files']:
            for query in json.loads(Path(entry['path']).read_text()):
                if 'source' not in query or 'destination' not in query: continue
                ids = [int(query['source']), int(query['destination'])]
                ix = np.searchsorted(coordinates[:, 0], ids)
                if np.any(ix >= n) or not np.array_equal(coordinates[ix, 0], ids): continue
                exposed.add(tuple(sorted(map(int, ix))))
        assert len(exposed) == cap['historical_exposure_unordered_od']
        reps = representatives(coordinates, box, mappings[128])
        pool = candidates(reps, reps, mappings[32], components, n, exposed)
        accepted_pool = np.load(parent/'p0'/f'{city}_hierarchical_grid_train_candidates.npz')['groups']
        accepted_keys = np.sort(accepted_pool[:, 0]*n+accepted_pool[:, 1])
        this_pool = pool[pool[:, 3] == 0]
        assert np.array_equal(np.sort(this_pool[:, 0]*n+this_pool[:, 1]), accepted_keys)
        del accepted_pool, accepted_keys, this_pool
        save_array(folder/'hierarchy_candidates.npy', pool)
        save_array(folder/'representatives128.npy', reps)
        budget = config['budgets'][city]; static = {}
        for name, role, points in (('sdc_Afit', 0, budget['Afit_points_per_FT']),
                                  ('Aselect', 1, budget['Aselect_points']),
                                  ('Acal', 2, budget['Acal_points']),
                                  ('mixed_test', 4, budget['mixed_test_points'])):
            eligible = pool[pool[:, 3] == role]
            groups = points//2; ordinary = groups*4//5; targeted = groups//5
            assert groups*2 == points and ordinary+targeted == groups and len(eligible) >= groups
            rng = np.random.default_rng(np.random.SeedSequence([config['sampler_seed'], city_code, role]))
            ordered = eligible[rng.permutation(len(eligible))]
            priority = change['scores'][mappings[64][ordered[ordinary:, 0]]]+change['scores'][mappings[64][ordered[ordinary:, 1]]]
            tail = ordered[ordinary:][np.argsort(-priority, kind='stable')]
            ordinary_ids = ordered[:ordinary, 0]*n+ordered[:ordinary, 1]
            target_ids = tail[:, 0]*n+tail[:, 1]
            save_array(folder/(name+'_ordinary_gids.npy'), ordinary_ids)
            save_array(folder/(name+'_target_order_gids.npy'), target_ids)
            static[name] = dict(role_code=role, legal_groups=len(eligible), required_groups=groups,
                                ordinary_groups=ordinary, required_targeted_groups=targeted,
                                target_candidates=len(target_ids), target_eligible_capacity='PENDING_GT_QUALIFICATION')
        accepted_plan = json.loads((parent/'data'/city/'plan.json').read_text())
        exact_uniform_capacity = int(sum(accepted_plan['candidate_role_capacity_by_d32'][0]))
        required = budget['Afit_points_per_FT']//2
        assert exact_uniform_capacity >= required
        rng = np.random.default_rng(np.random.SeedSequence([config['sampler_seed'], city_code, 99]))
        random_ids = uniform_fit(n, mappings[32], components, exposed, required, rng, role_codes)
        assert len(np.unique(random_ids)) == required
        save_array(folder/'random_Afit_gids.npy', random_ids)
        for q in paths: assert stamps[str(q)] == [q.stat().st_size, q.stat().st_mtime_ns]
        summary = dict(status='STATIC_PREFLIGHT_PASSED_TARGETS_PENDING', city=city, protocol=config,
            hierarchy_representatives=len(reps), hierarchy_legal_groups=len(pool), roles=static,
            random_fit_groups=required, random_exact_all_node_role0_capacity=exact_uniform_capacity,
            exact_capacity_source=str(parent/'data'/city/'plan.json'),
            capacity_reuse_basis='same complete node registry, reciprocal topology, bbox, roles and exposure exclusions',
            graph_policy_checks=graph_checks, detector=change['summary'], input_metadata=stamps,
            final_shared_holdout_frozen=False, formal_label_points=0, formal_training_updates=0,
            process_cpu_seconds=time.process_time()-cpu_started, elapsed_seconds=time.monotonic()-started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, no_hashes=True)
        atomic(report_path, summary); summaries.append(summary)
        old.lib.e1_free(old.handle); new.lib.e1_free(new.handle)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('protocol', 'input_metadata')}), flush=True)
    atomic(run/'static_preflight_summary.json', dict(status='STATIC_PREFLIGHT_PASSED_TARGETS_PENDING',
        results=summaries, final_shared_holdout_frozen=False, formal_label_points=0, formal_training_updates=0))


if __name__ == '__main__': main()
