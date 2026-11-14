"""One paid batch-one SSP observation per assigned query; frozen subset then full."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


ROOT = Path(os.environ['DIGO_E2E_RUN']).resolve()
PARENT = ROOT.parent
STAGES = [
    'input_prepare_seconds', 'h2d_and_gpu_rng_setup_seconds',
    'model_sampling_seconds', 'd2h_seconds', 'region_decode_union_seconds',
    'handoff_encode_seconds', 'handoff_receive_decode_seconds',
]
COUNTS = {'BJ_G0': 727, 'XA_G0': 872, 'BJ_G1': 1979, 'XA_G1': 4137}
FULL_COUNTS = {'BJ_G0': 6250, 'XA_G0': 3125, 'BJ_G1': 10000, 'XA_G1': 5000}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def rooted(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    assert path.resolve().is_relative_to(ROOT.resolve()), path
    return path


def e5_serial(job):
    value = job.get('e5_serial')
    return int(value) if value is not None else -1


def require_subset_complete():
    gate = json.loads((ROOT / 'SUBSET_READY_FOR_FULL.json').read_text())
    assert gate['status'] == 'ACCEPTED_SUBSET_COMPLETE' and gate['queries'] == 7715
    cost = json.loads((ROOT / 'COST_SUPPLEMENT_AUDIT_subset.json').read_text())
    assert cost['status'] == 'ACCEPTED_COMPLETE_SINGLETON_COMPONENT_ACCOUNTING'
    assert cost['scope'] == 'subset' and cost['queries'] == cost['records'] == 7715
    assert cost['unproven_region_changes'] == 0 and not cost['missing_worker_exits']
    search = json.loads((ROOT / 'audits/SUBSET_SEARCH_AUDIT.json').read_text())
    assert search['status'] == 'ACCEPTED_SUBSET_SINGLE_PASS'
    assert search.get('queries', search.get('groups')) == 7715


def jobs_contract():
    prepared = json.loads((ROOT / 'INPUT_ACCEPTANCE.json').read_text())
    assert prepared['status'] == 'ACCEPTED' and prepared['groups'] == 24375
    jobs = json.loads((ROOT / 'jobs.json').read_text())
    assert len(jobs) == len({int(j['serial']) for j in jobs}) == 24375
    all_jobs = {int(j['serial']): j for j in jobs}
    assert {key: sum(j['city'] + '_' + j['snapshot'] == key for j in jobs)
            for key in FULL_COUNTS} == FULL_COUNTS
    selected = {int(j['serial']): j for j in jobs if j['e5']}
    assert len(selected) == 7715
    counts = {key: sum(j['city'] + '_' + j['snapshot'] == key
                       for j in selected.values()) for key in COUNTS}
    assert counts == COUNTS, counts
    return all_jobs


def contracts():
    all_jobs = jobs_contract()
    subset = {s for s, j in all_jobs.items() if j['e5']}
    manifest = json.loads((ROOT / 'COST_MANIFEST.json').read_text())
    slots = manifest['slots']
    serials = [int(s) for slot in slots for s in slot['serials']]
    assert len(serials) == len(set(serials))
    members = set(serials)
    allowed = {'subset': subset, 'remainder': set(all_jobs) - subset, 'full': set(all_jobs)}
    matching_scopes = [scope for scope, wanted in allowed.items() if members == wanted]
    assert len(matching_scopes) == 1, ('Manifest must cover complete frozen subset, remainder or full workload', len(members))
    if 'scope' in manifest:
        assert manifest['scope'] == matching_scopes[0], manifest['scope']
    assert len({str(slot['worker']) for slot in slots}) == len(slots)
    assert len({str(rooted(slot['directory']).resolve()) for slot in slots}) == len(slots)
    for slot in slots:
        rooted(slot['directory'])
    return {s: all_jobs[s] for s in members}, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    selected, manifest = contracts()
    matches = [s for s in manifest['slots'] if str(s['worker']) == args.worker]
    assert len(matches) == 1, args.worker
    slot = matches[0]
    host = socket.gethostname().split('.')[0]
    assert host == slot['host']
    assert len(os.sched_getaffinity(0)) == 1
    out = rooted(slot['directory'])
    out.mkdir(parents=True, exist_ok=True)
    if 'E4_OUTPUT_DIR' in os.environ:
        assert Path(os.environ['E4_OUTPUT_DIR']).resolve() == out.resolve()
    lock = (out / 'cost.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assigned = [selected[int(s)] for s in slot['serials']]
    if any(not j['e5'] for j in assigned):
        require_subset_complete()
    assignment = dict(
        worker=slot['worker'], host=host, serials=slot['serials'],
        jobs=assigned, expected_records=len(assigned),
        method='SSP_singleton', batch=1, forward_conditions=1, nfe=1,
        scope='missing single-condition cost measurement only; no routing calls')
    if (out / 'assignment.json').exists():
        assert json.loads((out / 'assignment.json').read_text()) == assignment
    else:
        atomic(out / 'assignment.json', assignment)
    started = time.monotonic()
    completed = 0

    def state(status, **extra):
        atomic(out / 'progress.json', dict(
            status=status, worker=slot['worker'], pid=os.getpid(), host=host,
            saved=completed, expected=len(assigned), updated_unix=time.time(),
            elapsed_seconds=time.monotonic() - started, **extra))

    state('STARTING_SINGLE_CONDITION_COSTS')
    # Importing/model construction follows manifest checks and the exclusive lock.
    import torch
    sys.path.insert(0, str(PARENT / 'e1_v1/code'))
    import model64 as mt
    import evaluator64 as ev
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    gpu_name = torch.cuda.get_device_name()
    assert 'A100' in gpu_name
    gpu_uuid = os.environ.get('E4_SELECTED_GPU_UUID')
    props = torch.cuda.get_device_properties(0)
    visible_uuid = str(getattr(props, 'uuid', gpu_uuid or 'unavailable'))
    if gpu_uuid and visible_uuid.startswith('GPU-'):
        assert visible_uuid == gpu_uuid, (visible_uuid, gpu_uuid)
    with torch.no_grad():
        for city, snapshot in [('BJ', 'G0'), ('XA', 'G0'), ('BJ', 'G1'), ('XA', 'G1')]:
            case = [j for j in assigned if (j['city'], j['snapshot']) == (city, snapshot)]
            if not case:
                continue
            pending = []
            for job in case:
                target = out / 'cost_rows' / f'{int(job["serial"]):05d}_SSP_singleton.json'
                if target.exists():
                    prior = json.loads(target.read_text())
                    assert prior['job'] == job and prior['status'] == 'MEASURED'
                    assert prior['batch'] == prior['forward_conditions'] == prior['nfe'] == 1
                    completed += 1
                else:
                    claim = out / 'claims' / f'{int(job["serial"]):05d}.json'
                    assert not claim.exists(), ('Prior inference attempt has no completed record; inspect without repeating', str(claim))
                    pending.append(job)
            if not pending:
                state('REUSED_COMPLETE_COST_ROWS')
                continue
            role = 'base_test' if snapshot == 'G0' else 'mixed_test'
            ppg = 16 if snapshot == 'G0' else 2
            folder = PARENT / 'e3_v1/data' / city / snapshot / role
            odt = np.load(folder / 'odt.npy', mmap_mode='r')
            qids = np.load(folder / 'query_ids.npy', mmap_mode='r')
            support = np.load(folder / 'support.npy').astype(bool).reshape(4096)
            endpoints = np.load(folder / 'endpoints.npy', mmap_mode='r')
            input_folder = ROOT / 'inputs' / city / snapshot
            with np.load(input_folder / 'queries.npz') as inp:
                groups = np.array(inp['group_ids'])
                singletons = np.array(inp['singleton_packbits'])
            saved_scores = np.load(input_folder / 'single_scores.npy', mmap_mode='r')
            assert saved_scores.shape == (len(groups), 4096)
            assert singletons.shape == (len(groups), 512)
            modeldir = (PARENT / 'e1_v1/training' / f'{city}_hierarchical_grid_seed42'
                        if snapshot == 'G0' else
                        PARENT / 'e2_v1/training' / f'{city}_sdc_mixed_ft_seed42')
            checkpoint = modeldir / 'selected.pt'
            threshold = json.loads((modeldir / 'calibration.json').read_text())['threshold']
            setup_started = time.monotonic()
            checkpoint_stat = checkpoint.stat()
            model = mt.DiT().cuda()
            model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=False)['model'], strict=True)
            model.eval()
            torch.cuda.synchronize()
            setup_path = out / 'setup' / f'{city}_{snapshot}.json'
            if setup_path.exists():
                setup_path = out / 'setup' / f'{city}_{snapshot}_resume_{time.time_ns()}.json'
            atomic(setup_path, dict(
                seconds=time.monotonic() - setup_started, model_path=str(checkpoint),
                checkpoint_size=checkpoint_stat.st_size,
                checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns,
                threshold=threshold, batch=1, nfe=1, no_query_warmups=True,
                cuda_graph=False, compilation=False, setup_forward_images=0,
                timing_scope='setup excluded from per-query components; recorded separately'))
            for job in pending:
                serial = int(job['serial'])
                index = int(job['index'])
                gid = int(job['group_id'])
                assert int(groups[index]) == gid
                assert int(qids[index * ppg]) == gid * ppg
                target = out / 'cost_rows' / f'{serial:05d}_SSP_singleton.json'
                assert not target.exists()
                claim = out / 'claims' / f'{serial:05d}.json'
                claim.parent.mkdir(exist_ok=True)
                # A crash after a claim never silently triggers repeated inference.
                with claim.open('x') as handle:
                    json.dump(dict(serial=serial, worker=slot['worker'], pid=os.getpid(),
                                   claimed_unix=time.time(), no_automatic_retry=True), handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                state('MEASURING_COMPONENTS', current=dict(serial=serial, city=city, snapshot=snapshot))
                stages = {}
                row = dict(job=job, serial=serial, e5_serial=e5_serial(job), e5_subset=bool(job['e5']),
                           method='SSP_singleton', worker=slot['worker'], host=host,
                           gpu=gpu_name, gpu_uuid=gpu_uuid or visible_uuid,
                           requested_gpu=slot.get('gpu'),
                           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                           affinity=sorted(os.sched_getaffinity(0)), started_unix=time.time(),
                           model_path=str(checkpoint), threshold=threshold,
                           batch=1, forward_conditions=1, nfe=1, selected_condition_index=0,
                           no_search_calls=True, inter_OD_concurrency=1,
                           noise_role=role, noise_query_id=gid * ppg, eval_seed=20260912,
                           snapshot_id=f'{city}_{snapshot}_canonical_fifo_continuous',
                           setup_excluded=True, no_query_warmups=True, cuda_graph=False,
                           union_calls=0, region_operation='one-condition threshold/support/endpoints; no union')

                def measure(name, function, gpu=False):
                    if gpu:
                        torch.cuda.synchronize()
                    before = time.perf_counter()
                    value = function()
                    if gpu:
                        torch.cuda.synchronize()
                    stages[name] = time.perf_counter() - before
                    return value

                try:
                    assert checkpoint.stat().st_mtime_ns == checkpoint_stat.st_mtime_ns
                    pipeline_started = time.perf_counter()

                    def prepare():
                        condition = np.array(odt[index * ppg:index * ppg + 1])
                        endpoint = np.array(endpoints[index * ppg]).reshape(-1)
                        noise = ev.fixed_query_noise(
                            np.asarray([gid * ppg], dtype=np.int64), eval_seed=20260912,
                            city=city, snapshot_id=f'{city}_{snapshot}_canonical_fifo_continuous',
                            role=role, shape=(2, 64, 64))
                        assert condition.shape == (1, 5) and noise.shape == (1, 2, 64, 64)
                        return condition, endpoint, noise

                    condition, endpoint, noise = measure('input_prepare_seconds', prepare)
                    dcondition, dnoise = measure('h2d_and_gpu_rng_setup_seconds',
                        lambda: (torch.from_numpy(condition).cuda(), torch.from_numpy(noise).cuda()), True)
                    raw = measure('model_sampling_seconds',
                        lambda: mt.sample(model, dcondition, dnoise, nfe=1)[:, 0], True)
                    cpu = measure('d2h_seconds', lambda: raw.cpu().numpy(), True)

                    def decode():
                        scores = np.clip((cpu + np.float32(1)) / np.float32(2), 0, 1).reshape(4096)
                        assert scores.dtype == np.float32 and np.isfinite(scores).all()
                        mask = (scores.astype(np.float64) >= threshold) & support
                        mask[endpoint] = True
                        return scores, mask

                    scores, mask = measure('region_decode_union_seconds', decode)
                    payload = measure('handoff_encode_seconds', lambda: np.packbits(mask).tobytes())
                    delivered = measure('handoff_receive_decode_seconds',
                        lambda: np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).copy())
                    pipeline_wall = time.perf_counter() - pipeline_started
                    assert len(payload) == 512 and np.array_equal(delivered, mask)
                    assert set(stages) == set(STAGES)
                    expected = np.unpackbits(singletons[index]).astype(bool)
                    expected_score = np.asarray(saved_scores[index])
                    delta = float(np.max(np.abs(scores - expected_score)))
                    differences = int(np.count_nonzero(mask != expected))
                    # Preserve all fresh outputs before any binding decision.
                    scorepath = out / 'cost_scores' / f'{serial:05d}.npz'
                    scorepath.parent.mkdir(exist_ok=True)
                    temp = scorepath.with_suffix('.tmp.npz')
                    with temp.open('wb') as handle:
                        np.savez(handle, serial=serial, e5_serial=e5_serial(job), e5_subset=bool(job['e5']),
                                 group_id=gid, threshold=threshold, scores=scores,
                                 measured_packbits=np.packbits(mask),
                                 saved_packbits=singletons[index])
                        handle.flush()
                        os.fsync(handle.fileno())
                    temp.replace(scorepath)
                    core = sum(stages[k] for k in ['input_prepare_seconds', 'model_sampling_seconds', 'region_decode_union_seconds'])
                    transfer = stages['h2d_and_gpu_rng_setup_seconds'] + stages['d2h_seconds']
                    bridge = stages['handoff_encode_seconds'] + stages['handoff_receive_decode_seconds']
                    row.update(status='MEASURED', stage_seconds=stages,
                               inference_core_seconds=core, gpu_transfer_seconds=transfer,
                               cpu_interface_handoff_seconds=bridge,
                               inference_pipeline_excluding_cpu_handoff_seconds=core + transfer,
                               all_components_seconds=core + transfer + bridge,
                               instrumented_pipeline_wall_seconds=pipeline_wall,
                               scores_path=str(scorepath), score_max_abs_error=delta,
                               score_tolerance_exceeded=delta > 1e-5,
                               measured_singleton_packbits=np.packbits(mask).tolist(),
                               saved_singleton_difference_cells=differences,
                               binding_status=('EXACT_INITIAL_REGION' if differences == 0 else
                                               'NUMERICAL_REGION_DIFFERENCE_REQUIRES_PROOF'),
                               handoff_scope='in-process512-byte mask encode/decode; not network/IPC; Native argument/result conversion stays in search API')
                except Exception as exc:
                    row.update(status='FAILED', error=repr(exc), stage_seconds=stages)
                atomic(target, row)
                completed += 1
                state('COST_RECORD_SAVED')
                if row['status'] != 'MEASURED':
                    raise RuntimeError('Saved failed cost record; no automatic retry: ' + row['error'])
            del model, saved_scores, odt, qids, endpoints
            torch.cuda.empty_cache()
    state('COST_MEASUREMENT_FINISHED')


if __name__ == '__main__':
    main()
