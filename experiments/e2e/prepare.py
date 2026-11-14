"""Prepare the complete fixed-j0 ablation from accepted arrays; never infer or solve."""
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path(os.environ['DIGO_E2E_RUN']).resolve()
PARENT = ROOT.parent
E4 = PARENT / 'e4_full_v2'
E5 = PARENT / 'e5_speedup_gt5_v1'
ARM = 'SSP_single_GES'
PARENT_ARM = 'GES_deferred_shared_endpoint'
SETTINGS = [('BJ', 'G0', 6250), ('XA', 'G0', 3125),
            ('BJ', 'G1', 10000), ('XA', 'G1', 5000)]
E5_COUNTS = dict(BJ_G0=727, XA_G0=872, BJ_G1=1979, XA_G1=4137)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    with temporary.open('w') as handle:
        json.dump(value, handle, separators=(',', ':'))
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def stamp(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def assert_stamp(value):
    assert stamp(value['path']) == value, ('Changed saved source', value['path'])


def save_npz_once(path, arrays):
    if path.exists():
        with np.load(path) as previous:
            assert set(previous.files) == set(arrays)
            assert all(np.array_equal(previous[key], value)
                       for key, value in arrays.items()), str(path)
        return
    temporary = path.with_suffix('.tmp.npz')
    with temporary.open('wb') as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def prepare_setting(city, snapshot, count):
    started = time.monotonic()
    folder = ROOT / 'inputs' / city / snapshot
    folder.mkdir(parents=True, exist_ok=True)
    parent_queries = E4 / 'inputs' / city / snapshot / 'queries.npz'
    with np.load(parent_queries) as saved:
        queries = {key: saved[key] for key in saved.files}
    assert len(queries['group_ids']) == count
    assert queries['node_pairs'].shape == queries['node_indices'].shape == (count, 2)
    assert queries['initial_unions_packbits'].shape == (count, 512)
    assert queries['initial_unions_packbits'].dtype == np.uint8
    assert len(set(map(int, queries['group_ids']))) == count
    parent = PARENT / ('e1_v1' if snapshot == 'G0' else 'e2_v1')
    model_name = f'{city}_{"hierarchical_grid" if snapshot == "G0" else "sdc_mixed_ft"}_seed42'
    model = parent / 'training' / model_name
    calibration_path = model / 'calibration.json'
    threshold = json.loads(calibration_path.read_text())['threshold']
    scores_path = model / 'test_predictions' / 'scores.npy'
    query_ids_path = model / 'test_predictions' / 'query_ids.npy'
    scores = np.load(scores_path, mmap_mode='r')
    ids = np.load(query_ids_path, mmap_mode='r')
    points = 16 if snapshot == 'G0' else 2
    assert scores.shape == (count * points, 4096) and scores.dtype == np.float32
    assert np.array_equal(ids.reshape(count, points),
                          queries['group_ids'][:, None] * points + np.arange(points))
    coordinates_path = PARENT / 'e1_v1' / 'data' / city / 'coordinates.npy'
    cells_path = PARENT / 'e1_v1' / 'data' / city / 'cell64.npy'
    coordinates = np.load(coordinates_path)
    cells = np.load(cells_path)
    assert np.array_equal(coordinates[queries['node_indices'], 0], queries['node_pairs'])
    support = np.zeros(4096, dtype=bool)
    support[cells] = True
    endpoints = cells[queries['node_indices']]
    graph_npz = parent / 'data' / city / ('routing_graph.npz' if snapshot == 'G0' else 'G1_graph.npz')
    with np.load(graph_npz) as graph:
        pair, weights = np.unique(cells[graph['u']] * np.int64(4096) + cells[graph['v']],
                                  return_counts=True)
        original_arcs = len(graph['u'])
    assert int(weights.sum()) == original_arcs
    tail, head = pair // 4096, pair % 4096
    packed = np.empty((count, 512), dtype=np.uint8)
    allowed = np.empty(count, dtype=np.int64)
    equal = np.empty(count, dtype=bool)
    score_output = folder / 'single_scores.npy'
    temporary_scores = folder / 'single_scores.tmp.npy'
    if score_output.exists():
        selected_scores = np.load(score_output, mmap_mode='r')
        assert selected_scores.shape == (count, 4096) and selected_scores.dtype == np.float32
    else:
        selected_scores = np.lib.format.open_memmap(temporary_scores, mode='w+',
                                                   dtype=np.float32, shape=(count, 4096))
    source_stamps = [stamp(path) for path in [parent_queries, scores_path, query_ids_path,
                                             calibration_path, coordinates_path, cells_path, graph_npz]]
    max_parent_j0_score_error = 0.0
    all_parent_j0_scores_exact = True
    for begin in range(0, count, 64):
        end = min(count, begin + 64)
        chosen = np.asarray(scores[begin * points:end * points:points])
        assert chosen.shape == (end - begin, 4096)
        assert np.isfinite(chosen).all() and np.all((chosen >= 0) & (chosen <= 1))
        masks = (chosen.astype(np.float64) >= threshold) & support
        for offset, endpoints_row in enumerate(endpoints[begin:end]):
            masks[offset, endpoints_row] = True
        if snapshot == 'G1':
            chunk_path = E4 / 'predictions' / f'{city}_G1' / f'chunk_{begin:05d}_{end:05d}.npz'
            with np.load(chunk_path) as chunk:
                assert np.array_equal(chunk['group_ids'], queries['group_ids'][begin:end])
                assert np.array_equal(chunk['node_indices'], queries['node_indices'][begin:end])
                assert np.array_equal(chunk['departures'], np.arange(43200, 57600, 900))
                other = chunk['scores'][:, 0]
                other_masks = (other.astype(np.float64) >= threshold) & support
                for offset, endpoints_row in enumerate(endpoints[begin:end]):
                    other_masks[offset, endpoints_row] = True
                error = float(np.max(np.abs(chosen - other)))
                assert error <= 1e-5 and np.array_equal(masks, other_masks)
                max_parent_j0_score_error = max(max_parent_j0_score_error, error)
                all_parent_j0_scores_exact &= np.array_equal(chosen, other)
            source_stamps.append(stamp(chunk_path))
        unions = np.unpackbits(queries['initial_unions_packbits'][begin:end], axis=1).astype(bool)
        assert np.all(~masks | unions), (city, snapshot, begin, 'Singleton not a union subset')
        packed[begin:end] = np.packbits(masks, axis=1)
        allowed[begin:end] = ((masks[:, tail] & masks[:, head]) * weights).sum(axis=1)
        assert np.all(allowed[begin:end] <= queries['allowed_arcs'][begin:end])
        equal[begin:end] = np.all(packed[begin:end] == queries['initial_unions_packbits'][begin:end], axis=1)
        if score_output.exists():
            assert np.array_equal(selected_scores[begin:end], chosen)
        else:
            selected_scores[begin:end] = chosen
    if not score_output.exists():
        selected_scores.flush()
        del selected_scores
        temporary_scores.replace(score_output)
    else:
        del selected_scores
    arrays = dict(queries, singleton_packbits=packed, singleton_allowed_arcs=allowed,
                  singleton_initial_equal_union=equal)
    save_npz_once(folder / 'queries.npz', arrays)
    graph_text = folder / 'graph.txt'
    old_graph_text = E4 / 'inputs' / city / snapshot / 'graph.txt'
    if graph_text.is_symlink() or graph_text.exists():
        assert graph_text.resolve() == old_graph_text.resolve()
    else:
        graph_text.symlink_to(old_graph_text)
    for value in source_stamps:
        assert_stamp(value)
    report = dict(status='ACCEPTED', city=city, snapshot=snapshot, groups=count,
                  selected_condition=0, selected_departure=28800 if snapshot == 'G0' else 43200,
                  interval=[28800, 43200] if snapshot == 'G0' else [43200, 57600],
                  parent_points_per_group=points, threshold=threshold,
                  model_path=str(model / 'selected.pt'), parent_scores_path=str(scores_path),
                  parent_query_ids_path=str(query_ids_path), selected_scores_path=str(score_output),
                  queries_path=str(folder / 'queries.npz'), graph_text_path=str(graph_text),
                  singleton_subset_of_parent_union=True,
                  j0_masks_identical_to_e4=True, j0_scores_identical_to_e4=bool(all_parent_j0_scores_exact),
                  j0_max_score_difference=max_parent_j0_score_error,
                  identical_initial_masks=int(equal.sum()), original_directed_arcs=original_arcs,
                  original_arc_decisions_represented=count * original_arcs,
                  cell_pair_multiplicity_preserves_all_original_directed_arcs=True,
                  mean_initial_cells=float(np.unpackbits(packed, axis=1).sum(axis=1).mean()),
                  initial_edge_reduction_percent=100 * (1 - float(allowed.mean()) / original_arcs),
                  source_stamps=source_stamps,
                  output_stamps=[stamp(folder / 'queries.npz'), stamp(score_output)],
                  inference_calls=0, solver_calls=0, content_hashes=False,
                  elapsed_seconds=time.monotonic() - started)
    atomic(folder / 'acceptance.json', report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ['source_stamps', 'output_stamps']}), flush=True)
    return queries, equal, report


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'prepare.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    accepted = ROOT / 'INPUT_ACCEPTANCE.json'
    if accepted.exists():
        previous = json.loads(accepted.read_text())
        assert previous['status'] == 'ACCEPTED' and previous['groups'] == 24375
        for report in previous['settings']:
            for value in report['source_stamps'] + report['output_stamps']:
                assert_stamp(value)
        assert_stamp(previous['jobs_stamp'])
        print(json.dumps(dict(status='REUSED_COMPLETE_PREPARATION', groups=24375,
                              inference_calls=0, solver_calls=0, content_hashes=False)), flush=True)
        return
    assert json.loads((E4 / 'audits/FULL_SINGLE_PASS_AUDIT.json').read_text())['status'] == 'ACCEPTED_FULL_SINGLE_PASS'
    assert json.loads((E4 / 'INPUT_ACCEPTANCE.json').read_text())['status'] == 'ACCEPTED'
    assert json.loads((E4 / 'builds/formal_v2_r2/FREEZE.json').read_text())['status'] == 'FROZEN'
    manifest_path = E4 / 'RUN_MANIFEST_PARTS.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['repeats'] == 1
    mapping = {}
    for slot in manifest['slots']:
        for serial in slot['serials']:
            assert serial not in mapping
            mapping[serial] = slot
    assert set(mapping) == set(range(24375))
    e5_path = E5 / 'dataset/queries.json'
    e5 = json.loads(e5_path.read_text())
    assert len(e5) == 7715
    e5_map = {entry['serial']: (index, entry) for index, entry in enumerate(e5)}
    assert len(e5_map) == 7715
    settings = {}
    for city, snapshot, count in SETTINGS:
        settings[city, snapshot] = prepare_setting(city, snapshot, count)
    jobs = []
    counts = {key: 0 for key in E5_COUNTS}
    for serial in range(24375):
        slot = mapping[serial]
        audit_path = E4 / 'audits/single_pass_queries' / f'{serial:05d}.json'
        accepted_parent = json.loads(audit_path.read_text())
        assert accepted_parent['status'] == 'PASSED'
        record = accepted_parent['record']
        city, snapshot, index = record['city'], record['snapshot'], record['index']
        assert record['serial'] == serial and record['host'] == slot['host']
        queries, equal, setting = settings[city, snapshot]
        source_directory = Path(slot['directory']) / 'rows' / f'{serial:05d}'
        assert source_directory == Path(accepted_parent['source']['directory'])
        job = dict(serial=serial, city=city, snapshot=snapshot, index=index, host=slot['host'],
                   group_id=int(queries['group_ids'][index]),
                   source=int(queries['node_pairs'][index, 0]),
                   destination=int(queries['node_pairs'][index, 1]), interval=setting['interval'],
                   parent_rows_directory=str(source_directory), parent_audit_path=str(audit_path),
                   parent_row_path=str(source_directory / f'{PARENT_ARM}.json'),
                   parent_reference_path=str(source_directory / 'global_continuous_Astar.json'),
                   parent_noSSP_path=str(source_directory / 'noSSP_endpoint_corridor.json'),
                   identical_initial_mask=bool(equal[index]),
                   reuse_parent_search=bool(equal[index]),
                   singleton_source=dict(path=setting['selected_scores_path'], row=index,
                                         original_path=setting['parent_scores_path'],
                                         original_row=index * setting['parent_points_per_group'],
                                         query_id=int(queries['group_ids'][index]) * setting['parent_points_per_group'],
                                         condition=0, departure=setting['selected_departure'],
                                         threshold=setting['threshold']),
                   e5=serial in e5_map, e5_serial=serial if serial in e5_map else None,
                   e5_index=e5_map[serial][0] if serial in e5_map else None)
        if serial in e5_map:
            _, old = e5_map[serial]
            assert all(old[key] == job[key] for key in ['serial', 'city', 'snapshot', 'index', 'host', 'group_id', 'interval'])
            assert old['node_pair'] == [job['source'], job['destination']]
            counts[f'{city}_{snapshot}'] += 1
        jobs.append(job)
    assert counts == E5_COUNTS
    atomic(ROOT / 'jobs.json', jobs)
    atomic(accepted, dict(status='ACCEPTED', groups=24375, selected_point_predictions=24375,
                          e5_queries=7715, e5_counts=counts, settings=[value[2] for value in settings.values()],
                          frozen_condition=0, arm=ARM, method=4,
                          identical_initial_masks=sum(int(value[1].sum()) for value in settings.values()),
                          parent_manifest_stamp=stamp(manifest_path), e5_dataset_stamp=stamp(e5_path),
                          jobs_stamp=stamp(ROOT / 'jobs.json'),
                          reuse_rule='Exact initial-mask identity; same graph, OD, interval and method4',
                          inference_calls=0, solver_calls=0, content_hashes=False))
    print(json.dumps(dict(status='ACCEPTED', groups=24375, e5_queries=7715,
                          identical_initial_masks=sum(int(value[1].sum()) for value in settings.values()),
                          inference_calls=0, solver_calls=0, content_hashes=False)), flush=True)


if __name__ == '__main__':
    main()
