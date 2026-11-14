"""Exact saved-route replay and complete-cohort accounting; no model or solver calls."""
import argparse
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from prepare import ROOT, PARENT, E4, ARM, PARENT_ARM, SETTINGS, E5_COUNTS, atomic, stamp, assert_stamp


CACHE = {}


def error_bin(gap):
    if gap is None:
        return 'NO_PATH'
    if gap == 0:
        return '0'
    previous = 0
    for ceiling in [.1, .5, 1, 2, 5, 10]:
        if gap <= ceiling:
            return f'({previous:g},{ceiling:g}%]'
        previous = ceiling
    return '>10%'


def data(city, snapshot):
    key = city, snapshot
    if key not in CACHE:
        original = PARENT / ('e1_v1' if snapshot == 'G0' else 'e2_v1') / 'data' / city
        graph_path = original / ('routing_graph.npz' if snapshot == 'G0' else 'G1_graph.npz')
        coordinates_path = PARENT / 'e1_v1/data' / city / 'coordinates.npy'
        cells_path = PARENT / 'e1_v1/data' / city / 'cell64.npy'
        queries_path = ROOT / 'inputs' / city / snapshot / 'queries.npz'
        with np.load(graph_path) as saved:
            graph = {name: saved[name] for name in ['u', 'v', 'knots']}
        coordinates = np.load(coordinates_path)
        cells = np.load(cells_path)
        adjacency = [[] for _ in coordinates]
        for source, destination in zip(graph['u'], graph['v']):
            adjacency[int(source)].append(int(destination))
        with np.load(queries_path) as saved:
            queries = {name: saved[name] for name in saved.files}
        CACHE[key] = dict(graph=graph, coordinates=coordinates, cells=cells,
                          adjacency=adjacency, queries=queries,
                          stamps=[stamp(path) for path in [graph_path, coordinates_path,
                                                          cells_path, queries_path]])
    return CACHE[key]


def connected(source, destination, active, cells, adjacency):
    seen = {source}
    todo = [source]
    for node in todo:
        if node == destination:
            return True
        for next_node in adjacency[node]:
            if active[int(cells[next_node])] and next_node not in seen:
                seen.add(next_node)
                todo.append(next_node)
    return False


def parent_row(job, name, accepted_parent):
    path = Path(job['parent_rows_directory']) / f'{name}.json'
    metadata = next(value for value in accepted_parent['source']['files']
                    if value['name'] == path.name)
    actual = stamp(path)
    assert actual['size'] == metadata['size'] and actual['mtime_ns'] == metadata['mtime_ns'], str(path)
    row = json.loads(path.read_text())
    assert row['execution'] == 'COMPLETED' and row['host'] == job['host']
    assert row['group_id'] == job['group_id']
    assert [row['source'], row['destination']] == [job['source'], job['destination']]
    assert row['interval'] == job['interval'] and row['build'] == 'formal_v2_r2'
    identity = row['identity']
    assert all(identity[key] == job[key] for key in ['serial', 'city', 'snapshot', 'index'])
    assert identity['arm'] == name and identity['repeat'] == 0
    accepted_quality = accepted_parent['record']['arms'][name]
    if accepted_quality['valid']:
        assert row['result']['status'] == 'OK'
        assert row['result']['cost'] == accepted_quality['cost']
        assert row['result']['departure'] == accepted_quality['departure']
    assert row['native_seconds'] == accepted_quality['native_seconds'][0]
    assert row['api_seconds'] == accepted_quality['api_seconds'][0]
    return row, actual


def audit_row(path, job):
    """Audit one immutable row against original graph functions and accepted witnesses."""
    path = Path(path)
    row_source = stamp(path)
    row = json.loads(path.read_text())
    identity = row['identity']
    assert identity['arm'] == ARM and identity['repeat'] == 0
    assert all(identity[key] == job[key] for key in ['serial', 'city', 'snapshot', 'index'])
    assert row['host'] == job['host'] and row['group_id'] == job['group_id']
    assert [row['source'], row['destination']] == [job['source'], job['destination']]
    assert row['interval'] == job['interval'] and row['singleton_source'] == job['singleton_source']
    assert row['build'] == 'formal_v2_r2' and len(row['affinity']) == 1
    if row['execution'] != 'COMPLETED':
        return dict(status='EXECUTION_FAILED', serial=job['serial'], job=job,
                    row_path=str(path), result_path=str(path), source=row_source,
                    execution=row['execution'], error=row.get('error'), valid=False,
                    quality=dict(valid=False, cost_exact=False, earliest_exact=False,
                                 gap_percent=None, error_bin='EXECUTION_FAILED'),
                    solver_calls=0, inference_calls=0, content_hashes=False)
    entry = data(job['city'], job['snapshot'])
    graph, coordinates, cells = entry['graph'], entry['coordinates'], entry['cells']
    queries, index = entry['queries'], job['index']
    source, destination = map(int, queries['node_indices'][index])
    assert queries['node_pairs'][index].tolist() == [job['source'], job['destination']]
    assert int(queries['group_ids'][index]) == job['group_id']
    initial = np.unpackbits(queries['singleton_packbits'][index]).astype(bool)
    union = np.unpackbits(queries['initial_unions_packbits'][index]).astype(bool)
    assert initial.shape == (4096,) and np.all(~initial | union)
    assert initial[int(cells[source])] and initial[int(cells[destination])]
    assert bool(np.array_equal(initial, union)) == job['identical_initial_mask']
    accepted_parent = json.loads(Path(job['parent_audit_path']).read_text())
    assert accepted_parent['status'] == 'PASSED'
    assert accepted_parent['source']['directory'] == job['parent_rows_directory']
    union_row, union_stamp = parent_row(job, PARENT_ARM, accepted_parent)
    reference, reference_stamp = parent_row(job, 'global_continuous_Astar', accepted_parent)
    result = row['result']
    assert result['method'] == 4 and result['solver'] == 'GES-AStar-full-v2'
    assert math.isfinite(row['api_seconds']) and math.isfinite(row['native_seconds'])
    assert 0 <= row['native_seconds'] <= row['api_seconds']
    assert row['native_seconds'] == result['native_seconds']
    assert row['stage_seconds'] == result['stage_seconds']
    initial_connected = connected(source, destination, initial, cells, entry['adjacency'])
    assert result['initial_connected'] == initial_connected
    assert result['repaired'] == (not initial_connected)
    active = np.zeros(4096, dtype=bool)
    active[result['active_cells']] = True
    assert len(set(result['active_cells'])) == len(result['active_cells'])
    assert initial[int(cells[source])] and initial[int(cells[destination])]
    expected_active = initial.copy()
    source_stamps = [union_stamp, reference_stamp]
    bridge_path = None
    if result['repaired']:
        bridge, bridge_stamp = parent_row(job, 'noSSP_endpoint_corridor', accepted_parent)
        assert bridge['result']['method'] == 6 and bridge['result']['repaired']
        assert bridge['result']['status'] == 'OK'
        expected_active[bridge['result']['active_cells']] = True
        source_stamps.append(bridge_stamp)
        bridge_path = bridge_stamp['path']
    assert np.array_equal(active, expected_active), 'Unexpected endpoint-repair region'
    assert active[int(cells[source])] and active[int(cells[destination])]
    assert all(0 <= unique <= scans for unique, scans in zip(result['stage_unique'], result['stage_scans']))
    assert len(result['stage_unique']) == len(result['stage_scans']) == len(result['stage_seconds']) == 5
    assert all(math.isfinite(value) and value >= 0 for value in result['stage_seconds'])
    assert 0 <= result['edge_unique_all'] <= len(graph['u'])
    valid = result['status'] == 'OK'
    quality = dict(valid=valid, cost=None, departure=None, cost_exact=False, earliest_exact=False,
                   gap_percent=None, absolute_gap_seconds=None, zero_reference_positive_cost=False)
    if valid:
        assert len(result['edge_ids']) + 1 == len(result['route']) == len(result['arrivals'])
        assert result['route'][0] == job['source'] and result['route'][-1] == job['destination']
        departure = at = F(result['departure'])
        assert job['interval'][0] <= at <= job['interval'][1]
        for step, edge in enumerate(result['edge_ids']):
            assert 0 <= edge < len(graph['u'])
            tail, head = int(graph['u'][edge]), int(graph['v'][edge])
            knots = graph['knots'][edge]
            assert result['route'][step:step + 2] == [int(coordinates[tail, 0]), int(coordinates[head, 0])]
            assert active[int(cells[tail])] and active[int(cells[head])]
            assert at == F(result['arrivals'][step])
            slot = int(at // 900)
            remainder = at - slot * 900
            at += int(knots[slot % 96]) + F(int(knots[(slot + 1) % 96]) - int(knots[slot % 96]), 900) * remainder
        assert at == F(result['arrivals'][-1]) and at - departure == F(result['cost'])
        assert len(set(result['edge_ids'])) <= result['edge_unique_all']
        cost, reference_cost = F(result['cost']), F(reference['result']['cost'])
        assert cost >= reference_cost
        gap = float((cost - reference_cost) * 100 / reference_cost) if reference_cost else (0.0 if cost == 0 else None)
        quality.update(cost=result['cost'], departure=result['departure'], cost_exact=cost == reference_cost,
                       earliest_exact=cost == reference_cost and departure == F(reference['result']['departure']),
                       gap_percent=gap, absolute_gap_seconds=float(cost - reference_cost),
                       zero_reference_positive_cost=reference_cost == 0 and cost > 0,
                       exact_original_edge_replay=True)
    else:
        assert result['status'] == 'NO_PATH'
        assert not connected(source, destination, active, cells, entry['adjacency'])
        quality['no_path_independently_verified'] = True
    old = union_row['result']
    assert not (initial_connected and not old['initial_connected']), 'Subset connectivity contradiction'
    repair_stratum = ('both_connected' if initial_connected else
                      'both_disconnected' if not old['initial_connected'] else
                      'single_disconnected_union_connected')
    if initial_connected == old['initial_connected']:
        assert set(result['active_cells']) <= set(old['active_cells'])
        if valid:
            assert F(result['cost']) >= F(old['cost']), 'Subset region improved exact optimum'
    reused = bool(row.get('reused_without_solving'))
    assert reused == job['reuse_parent_search']
    if reused:
        assert job['identical_initial_mask']
        assert row['reuse_source'] == str(Path(job['parent_row_path']))
        assert row['result'] == union_row['result']
        assert row['api_seconds'] == union_row['api_seconds']
        assert row['native_seconds'] == union_row['native_seconds']
        assert row['stage_seconds'] == union_row['stage_seconds']
        assert row['affinity'] == union_row['affinity']
        assert row['reuse_source_stamp'] == union_stamp
    else:
        assert not job['identical_initial_mask']
        attempt = json.loads(Path(row['attempt_path']).read_text())
        assert attempt['serial'] == job['serial'] and attempt['solver_calls_authorized'] == 1
    quality.update(error_bin=error_bin(quality['gap_percent']),
                   edge_percent=100 * result['edge_unique_all'] / len(graph['u']),
                   initial_edge_reduction_percent=100 * (1 - int(queries['singleton_allowed_arcs'][index]) / len(graph['u'])),
                   initial_connected=result['initial_connected'], repaired=result['repaired'],
                   active_cells_count=len(result['active_cells']),
                   initial_cells_count=int(initial.sum()), parent_initial_connected=old['initial_connected'],
                   parent_repaired=old['repaired'], repair_stratum=repair_stratum,
                   parent_cost=old.get('cost'), parent_departure=old.get('departure'),
                   api_seconds=row['api_seconds'], native_seconds=row['native_seconds'],
                   reused_without_solving=reused, optimal_cost=quality['cost_exact'],
                   optimal_cost_and_earliest=quality['earliest_exact'])
    assert stamp(path) == row_source, 'Result changed during audit'
    return dict(status='PASSED', serial=job['serial'], city=job['city'], snapshot=job['snapshot'],
                index=job['index'], group_id=job['group_id'], job=job, quality=quality, **quality,
                row_path=str(path), result_path=str(path), source=row_source,
                input_stamps=entry['stamps'], parent_source_stamps=source_stamps,
                parent_bridge_path=bridge_path, parent_row_path=job['parent_row_path'],
                verified_unix=time.time(), solver_calls=0, inference_calls=0, content_hashes=False)


def cached_audit(path, result_path, job):
    """Use an accepted exact replay only while every recorded dependency is unchanged."""
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('job') != job or value.get('source') != stamp(result_path):
        return None
    if value.get('status') == 'PASSED':
        for source in value['input_stamps'] + value['parent_source_stamps']:
            assert_stamp(source)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--scope', choices=['full', 'subset'], default='full')
    args = parser.parse_args()
    output = args.output_dir or ROOT / 'audits'
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    assert json.loads((ROOT / 'INPUT_ACCEPTANCE.json').read_text())['status'] == 'ACCEPTED'
    manifest = json.loads((ROOT / 'RUN_MANIFEST.json').read_text())
    all_jobs = json.loads((ROOT / 'jobs.json').read_text())
    assert len(all_jobs) == 24375 and [job['serial'] for job in all_jobs] == list(range(24375))
    jobs = all_jobs if args.scope == 'full' else [job for job in all_jobs if job['e5']]
    expected_queries = 24375 if args.scope == 'full' else 7715
    expected_counts = {f'{city}_{snapshot}': count for city, snapshot, count in SETTINGS} if args.scope == 'full' else E5_COUNTS
    assert len(jobs) == expected_queries
    selected_serials = {job['serial'] for job in jobs}
    mapping = {}
    for slot in manifest['slots']:
        for serial in slot['serials']:
            assert serial not in mapping and all_jobs[serial]['host'] == slot['host']
            mapping[serial] = slot
    assert set(mapping) == set(range(24375))
    audited, missing, errors, records = 0, [], [], []
    workers, newly_replayed = [], 0
    counts = {f'{city}_{snapshot}': 0 for city, snapshot, _ in SETTINGS}
    reused = computed = 0
    (output / 'per_query').mkdir(exist_ok=True)
    for slot in manifest['slots']:
        if not selected_serials.intersection(slot['serials']):
            continue
        if args.scope == 'subset':
            assert slot.get('cohort') == 'subset', 'Subset audit requires dedicated subset slots'
            assert set(slot['serials']) <= selected_serials, 'Subset/remainder may not share a slot'
        directory = Path(slot['directory'])
        present_rows = {int(path.stem) for path in (directory / 'rows').glob('*.json')
                        if not path.name.endswith('.tmp.json')}
        assert present_rows <= set(slot['serials']), ('Unassigned result rows', slot['worker'])
        attempts = {int(path.stem) for path in (directory / 'attempts').glob('*.json')
                    if not path.name.endswith('.tmp.json')}
        assert attempts <= set(slot['serials']), ('Unassigned query attempts', slot['worker'])
        assert not any(all_jobs[serial]['reuse_parent_search'] for serial in attempts), 'A reused query was attempted again'
        exit_path = directory / 'exit.json'
        exit_value = json.loads(exit_path.read_text()) if exit_path.exists() else None
        worker_exit_path = directory / 'worker_exit.json'
        worker_exit = json.loads(worker_exit_path.read_text()) if worker_exit_path.exists() else None
        finished = bool(exit_value and exit_value.get('exit_code') == 0 and
                        worker_exit and worker_exit.get('exit_code') == 0)
        workers.append(dict(worker=slot['worker'], host=slot['host'], directory=str(directory),
                            finished=finished, exit_code=exit_value.get('exit_code') if exit_value else None,
                            worker_exit_code=worker_exit.get('exit_code') if worker_exit else None))
    for job in jobs:
        slot = mapping[job['serial']]
        directory = Path(slot['directory'])
        result_path = directory / 'rows' / f'{job["serial"]:05d}.json'
        target = output / 'per_query' / result_path.name
        if not result_path.exists():
            missing.append(job['serial'])
            continue
        try:
            value = cached_audit(target, result_path, job)
            target_audit_current = value is not None
            if value is None:
                value = cached_audit(directory / 'audits' / result_path.name, result_path, job)
            if value is None:
                value = audit_row(result_path, job)
                newly_replayed += 1
            if value['status'] != 'PASSED':
                errors.append(dict(serial=job['serial'], status=value['status'],
                                   error=value.get('error'), row_path=str(result_path)))
                atomic(target, value)
                continue
            if not target_audit_current:
                atomic(target, value)
            counts[f'{job["city"]}_{job["snapshot"]}'] += 1
            audited += 1
            reused += int(value['reused_without_solving'])
            computed += int(not value['reused_without_solving'])
            records.append(dict(serial=job['serial'], job=job, quality=value['quality'],
                                result_path=str(result_path), audit_path=str(target)))
        except Exception as exception:
            errors.append(dict(serial=job['serial'], status='AUDIT_FAILED',
                               error=repr(exception), row_path=str(result_path)))
    complete = audited == expected_queries and counts == expected_counts and not errors and not missing and all(worker['finished'] for worker in workers)
    accepted_status = 'ACCEPTED_FULL_SINGLE_PASS' if args.scope == 'full' else 'ACCEPTED_SUBSET_SINGLE_PASS'
    report = dict(status=accepted_status if complete else 'PARTIAL_OR_FAILED', scope=args.scope,
                  queries=expected_queries, groups=expected_queries, expected_queries=expected_queries, audited_queries=audited,
                  expected_method_records=expected_queries, audited_method_records=audited,
                  expected_workers=len(workers), finished_workers=sum(worker['finished'] for worker in workers),
                  by_setting=counts, missing=missing, errors=errors, workers=workers,
                  parent_rows_reused=reused, new_search_rows=computed,
                  e5_queries=sum(bool(record['job']['e5']) for record in records),
                  newly_replayed_rows=newly_replayed, solver_calls=0, inference_calls=0,
                  content_hashes=False, observed_unix=time.time(),
                  elapsed_seconds=time.monotonic() - started)
    atomic(output / ('audited_queries.json' if args.scope == 'full' else 'audited_queries_subset.json'), records)
    atomic(output / ('FULL_SEARCH_AUDIT.json' if args.scope == 'full' else 'SUBSET_SEARCH_AUDIT.json'), report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ['workers', 'missing', 'errors']}), flush=True)
    if errors:
        print(json.dumps(dict(errors=errors[:20], total_errors=len(errors))), flush=True)


if __name__ == '__main__':
    main()
