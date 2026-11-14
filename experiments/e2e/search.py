"""One fixed-j0 SSP+GES call per OD, with durable no-repeat receipts and exact replay."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from prepare import ROOT, E4, ARM, PARENT_ARM, atomic, stamp
from audit_search import audit_row, cached_audit

sys.path.insert(0, str(E4 / 'code'))
from native import Native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'RUN_MANIFEST.json').read_text())
    slot = next(value for value in manifest['slots'] if value['worker'] == args.worker)
    host = socket.gethostname().split('.')[0]
    assert host == slot['host'] and len(os.sched_getaffinity(0)) == 1
    output = Path(os.environ['E4_OUTPUT_DIR'])
    assert output == Path(slot['directory'])
    assert json.loads((ROOT / 'INPUT_ACCEPTANCE.json').read_text())['status'] == 'ACCEPTED'
    assert json.loads((E4 / 'builds/formal_v2_r2/FREEZE.json').read_text())['status'] == 'FROZEN'
    lock = (output / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    all_jobs = json.loads((ROOT / 'jobs.json').read_text())
    assert len(all_jobs) == 24375
    jobs = [all_jobs[serial] for serial in slot['serials']]
    assert len({job['serial'] for job in jobs}) == len(jobs)
    assert all(job['host'] == host for job in jobs)
    identity = dict(worker=args.worker, host=host, affinity=sorted(os.sched_getaffinity(0)),
                    protocol='SSP_single_GES_fixed_j0_v1', repeats=1, arm=ARM,
                    method=4, build='formal_v2_r2', assigned_serials=slot['serials'],
                    assigned_queries=len(jobs), condition=0)
    if (output / 'identity.json').exists():
        previous = json.loads((output / 'identity.json').read_text())
        assert all(previous[key] == identity[key]
                   for key in ['worker', 'host', 'protocol', 'repeats', 'arm', 'method',
                               'build', 'assigned_serials', 'assigned_queries', 'condition'])
    else:
        atomic(output / 'identity.json', identity)
    atomic(output / 'assignment.json', dict(identity=identity, jobs=jobs))
    for name in ['rows', 'audits', 'attempts', 'setup']:
        (output / name).mkdir(exist_ok=True)
    started = time.monotonic()
    saved = computed = reused = skipped = errors = 0
    current = None
    cache = {}

    def state(status, **extra):
        atomic(output / 'progress.json', dict(status=status, pid=os.getpid(), **identity,
               current=current, saved=saved, computed=computed, reused=reused, skipped=skipped,
               errors=errors, updated_unix=time.time(), elapsed_seconds=time.monotonic() - started,
               process_cpu_seconds=time.process_time(), **extra))

    state('STARTING')
    try:
        for job in jobs:
            current = dict(serial=job['serial'], city=job['city'], snapshot=job['snapshot'], index=job['index'])
            target = output / 'rows' / f'{job["serial"]:05d}.json'
            audit_path = output / 'audits' / target.name
            if target.exists():
                # An error is a saved attempt as well. Never silently repeat it.
                audit = cached_audit(audit_path, target, job)
                if audit is None:
                    try:
                        audit = audit_row(target, job)
                    except Exception as exception:
                        audit = dict(status='AUDIT_FAILED', serial=job['serial'], job=job,
                                     source=stamp(target), row_path=str(target), error=repr(exception))
                    atomic(audit_path, audit)
                skipped += 1
                saved += 1
                errors += int(audit['status'] != 'PASSED')
                state('EXISTING_ROW_RETAINED')
                continue
            key = job['city'], job['snapshot']
            if key not in cache:
                with np.load(ROOT / 'inputs' / job['city'] / job['snapshot'] / 'queries.npz') as source:
                    queries = {name: source[name] for name in source.files}
                cache[key] = dict(queries=queries, handle=None)
            entry = cache[key]
            queries = entry['queries']
            index = job['index']
            assert int(queries['group_ids'][index]) == job['group_id']
            assert queries['node_pairs'][index].tolist() == [job['source'], job['destination']]
            mask = np.unpackbits(queries['singleton_packbits'][index])
            identical = np.array_equal(queries['singleton_packbits'][index],
                                       queries['initial_unions_packbits'][index])
            assert identical == job['identical_initial_mask'] == job['reuse_parent_search']
            row = dict(identity=dict(**current, repeat=0, arm=ARM, position=0),
                       worker=args.worker, host=host, affinity=identity['affinity'],
                       group_id=job['group_id'], source=job['source'], destination=job['destination'],
                       interval=job['interval'], build='formal_v2_r2',
                       singleton_source=job['singleton_source'], condition=0,
                       singleton_packbits=queries['singleton_packbits'][index].tolist(),
                       parent_row_path=job['parent_row_path'], e5=job['e5'],
                       started_unix=time.time(), reused_without_solving=bool(identical),
                       single_pass_executor='SSP_single_GES_v1')
            if identical:
                old = json.loads(Path(job['parent_row_path']).read_text())
                assert old['execution'] == 'COMPLETED' and old['host'] == host
                assert old['group_id'] == job['group_id'] and old['interval'] == job['interval']
                assert old['result']['method'] == 4
                row.update(execution='COMPLETED', result=old['result'],
                           api_seconds=old['api_seconds'], native_seconds=old['native_seconds'],
                           stage_seconds=old['stage_seconds'], affinity=old['affinity'],
                           reuse_source=job['parent_row_path'], reuse_source_stamp=stamp(job['parent_row_path']),
                           measurement_started_unix=old.get('started_unix'),
                           timing_provenance='Accepted original method4 observation; identical initial mask, no new call')
                reused += 1
            else:
                attempt_path = output / 'attempts' / target.name
                if attempt_path.exists():
                    row.update(execution='INCOMPLETE_PRIOR_ATTEMPT', api_seconds=None,
                               native_seconds=None, attempt_path=str(attempt_path),
                               error='A prior durable query-attempt receipt has no result; no automatic repeat')
                else:
                    if entry['handle'] is None:
                        state('LOADING_GRAPH')
                        before = time.monotonic()
                        entry['handle'] = Native(E4 / 'inputs' / job['city'] / job['snapshot'] / 'graph.txt',
                                                 E4 / 'builds/formal_v2_r2/libges_astar_full_v2.so')
                        atomic(output / 'setup' / f'{job["city"]}_{job["snapshot"]}.json',
                               dict(graph_load_seconds=time.monotonic() - before,
                                    excluded_from_query_timing=True, warmup_queries=0,
                                    graph_path=str(E4 / 'inputs' / job['city'] / job['snapshot'] / 'graph.txt'),
                                    library_source=stamp(E4 / 'builds/formal_v2_r2/libges_astar_full_v2.so')))
                    # Exclusive creation remains after interrupted calls, preventing blind retries.
                    attempt = dict(serial=job['serial'], host=host, pid=os.getpid(),
                                   source=job['source'], destination=job['destination'],
                                   interval=job['interval'], method=4, condition=0,
                                   solver_calls_authorized=1, created_unix=time.time())
                    with attempt_path.open('x') as handle:
                        json.dump(attempt, handle)
                        handle.flush()
                        os.fsync(handle.fileno())
                    row['attempt_path'] = str(attempt_path)
                    state('SEARCHING')
                    before = time.monotonic()
                    try:
                        result = entry['handle'].query(job['source'], job['destination'],
                                                       *job['interval'], 4, mask)
                        row.update(execution='COMPLETED', api_seconds=time.monotonic() - before,
                                   result=result, native_seconds=result['native_seconds'],
                                   stage_seconds=result['stage_seconds'])
                    except Exception as exception:
                        row.update(execution='NATIVE_ERROR', api_seconds=time.monotonic() - before,
                                   native_seconds=None, error=repr(exception))
                    computed += 1
            atomic(target, row)  # Persist the result, including errors, before any audit.
            try:
                audit = audit_row(target, job)
            except Exception as exception:
                audit = dict(status='AUDIT_FAILED', serial=job['serial'], job=job,
                             source=stamp(target), row_path=str(target), error=repr(exception))
            atomic(audit_path, audit)
            errors += int(audit['status'] != 'PASSED')
            saved += 1
            state('ROW_SAVED')
            if saved % 16 == 0 or saved == len(jobs):
                print(json.dumps(dict(worker=args.worker, saved=saved, expected=len(jobs),
                                      computed=computed, reused=reused, skipped=skipped, errors=errors)), flush=True)
        status = 'FINISHED' if errors == 0 else 'FINISHED_WITH_ERRORS'
        state(status, all_assigned_attempted=True)
        atomic(output / 'worker_exit.json', dict(status=status, exit_code=int(errors != 0),
               worker=args.worker, saved=saved, assigned=len(jobs), computed=computed,
               reused=reused, skipped=skipped, errors=errors, finished_unix=time.time(),
               inference_calls=0, content_hashes=False))
        return int(errors != 0)
    except BaseException as exception:
        state('FAILED', error=repr(exception))
        atomic(output / 'worker_exit.json', dict(status='FAILED', exit_code=1, worker=args.worker,
               saved=saved, computed=computed, reused=reused, skipped=skipped, errors=errors,
               current=current, error=repr(exception), finished_unix=time.time(),
               incomplete_attempts_are_never_automatically_repeated=True))
        raise
    finally:
        for entry in cache.values():
            if entry['handle'] is not None:
                entry['handle'].close()


if __name__ == '__main__':
    raise SystemExit(main())
