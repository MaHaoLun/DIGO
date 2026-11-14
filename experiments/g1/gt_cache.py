"""Resumable two-snapshot, two-departure native GT with paid-call ledger."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import fcntl
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

import h5py
import numpy as np

DEPARTURES = (43200, 44100)
FLAG_NAMES = ('shifted_pair_hit', 'changed_edge_at_entry', 'internal_hotspot_hit',
              'endpoint_only_hit', 'old_new_path_changed', 'duration_changed', 'affected')


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def atomic_array(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npy')
    with temporary.open('wb') as out: np.save(out, value)
    temporary.replace(path)


def evaluate_weights(weights, edges, times):
    phase = np.remainder(times, 86400.)
    slot = (phase//900).astype(np.int64); fraction = (phase-slot*900)/900
    left, right = weights[edges, slot], weights[edges, (slot+1) % 96]
    return left+(right-left)*fraction


def point_evidence(old, new, cells, hotmask, shifted, old_path, old_cost, path, arrival, cost):
    inside = np.unique(cells[path[1:-1]])
    internal_hot = inside[hotmask[inside]]
    endpoint = bool(np.any(hotmask[cells[path[[0, -1]]]]))
    edges = np.searchsorted(new.keys, path[:-1].astype(np.int64)*new.n+path[1:])
    significant = shifted[edges]
    at_entry = False
    if significant.any():
        selected = edges[significant]; times = arrival[:-1][significant]
        at_entry = bool(np.any(evaluate_weights(old.knots, selected, times) !=
                               evaluate_weights(new.knots, selected, times)))
    path_changed = not np.array_equal(old_path, path)
    internal = bool(len(internal_hot))
    flags = [bool(significant.any()), at_entry, internal, endpoint and not internal,
             path_changed, bool(old_cost != cost), at_entry or internal or path_changed]
    return np.asarray(flags, bool), internal_hot.astype(np.uint16)


class Cache:
    def __init__(self, run, city, workers=8):
        self.run, self.city, self.workers = Path(run), city, workers
        self.config = json.loads((self.run/'protocol.json').read_text())
        self.parent = Path(self.config['parent_e1_run']); self.root = Path(self.config['remote_root'])
        assert json.loads((self.parent/'E1_ACCEPTANCE.json').read_text())['status'] == 'ACCEPTED'
        assert json.loads((self.run/'audits/static_preflight_validation.json').read_text())['status'] == 'PASSED'
        self.folder = self.run/'data'/city
        self.preflight = json.loads((self.folder/'static_preflight.json').read_text())
        assert self.preflight['protocol'] == self.config
        self.check_inputs()
        self.lock = (self.folder/'gt_worker.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0, str(self.parent/'code'))
        from point_data import Native
        self.native = {stage:Native(self.root, city, stage) for stage in ('G0', 'G1')}
        self.n = self.native['G1'].n
        for stage, native in self.native.items():
            with np.load(self.folder/(stage+'_graph.npz')) as frozen:
                assert np.array_equal(native.u, frozen['u']) and np.array_equal(native.v, frozen['v'])
                assert np.array_equal(native.knots, frozen['knots'])
        self.cells = np.load(self.folder/'cell64.npy')
        with np.load(self.folder/'detection.npz') as detection:
            self.shifted = detection['shifted_arcs']
            self.hotmask = np.zeros(4096, bool); self.hotmask[detection['hotspots']] = True
        self.cache = self.run/'cache'/city; self.cache.mkdir(parents=True, exist_ok=True)
        self.index = {}
        for file in sorted(self.cache.glob('*/s*.h5')):
            if '.tmp.' not in file.name: self.register(file)
        self.stop = threading.Event(); self.event_lock = threading.Lock(); self.call_count = 0
        self.attempt = str(time.time_ns())+'_'+str(os.getpid())
        ledger = self.run/'ledger'/city; ledger.mkdir(parents=True, exist_ok=True)
        self.events = (ledger/('calls_'+self.attempt+'.jsonl')).open('a', buffering=1)
        self.started, self.cpu_started = time.monotonic(), time.process_time()

    def check_inputs(self):
        for name, expected in self.preflight['input_metadata'].items():
            p = Path(name); assert [p.stat().st_size, p.stat().st_mtime_ns] == expected, name

    def event(self, value):
        with self.event_lock:
            value = dict(attempt=self.attempt, unix=time.time(), **value)
            self.events.write(json.dumps(value)+'\n'); self.events.flush()

    def query(self, stage, source, departure, targets, block):
        if self.stop.is_set(): raise InterruptedError('GT generation interrupted before next query')
        with self.event_lock:
            self.call_count += 1; call_id = self.attempt+':'+str(self.call_count)
        self.event(dict(event='CALL_INTENT', call_id=call_id, stage=stage, source=int(source),
                        departure=departure, block=block, requested_point_paths=len(targets)))
        started, cpu = time.monotonic(), time.thread_time()
        result = self.native[stage].query(source, departure, targets)
        self.event(dict(event='CALL_COMPLETE', call_id=call_id, completed_point_paths=len(targets),
                        path_nodes=len(result[1]), elapsed_seconds=time.monotonic()-started,
                        thread_cpu_seconds=time.thread_time()-cpu))
        return result

    def register(self, file):
        with h5py.File(file, 'r') as f:
            assert f.attrs['status'] == 'COMPLETE' and f.attrs['city'] == self.city
            assert f.attrs['points_per_group'] == 2 and f.attrs['grid_dim'] == 64
            gids = f['group_ids'][:]
            assert len(np.unique(gids)) == len(gids) and np.all(gids[1:] > gids[:-1])
            for row, gid in enumerate(gids):
                key = int(gid)
                if key in self.index: assert self.index[key] == (str(file), row)
                self.index[key] = (str(file), row)
        return gids

    def source_job(self, block, source, gids):
        target = self.cache/block/f's{source:07d}.h5'
        assert not target.exists()
        targets = np.asarray(gids % self.n, np.int32)
        assert np.all(gids//self.n == source)
        routed, witnesses = {}, []; max_error = 0.
        started = time.monotonic()
        for stage in ('G0', 'G1'):
            native = self.native[stage]; calls = []
            for departure in DEPARTURES:
                off, nodes, arrival, costs = self.query(stage, source, departure, targets, block)
                max_error = max(max_error, native.replay(source, departure, targets, off, nodes, arrival, costs))
                exact = native.rational_replay(nodes[off[0]:off[1]], departure)
                assert abs(float(exact)-arrival[off[1]-1]) < 1e-7
                witnesses.append(dict(stage=stage, departure=departure, target=int(targets[0]),
                                      numerator=str(exact.numerator), denominator=str(exact.denominator)))
                calls.append((off, nodes, arrival, costs))
            routed[stage] = calls
        flags, hotcells, hotoff = [], [], [0]
        image_cells, image_progress, imageoff = [], [], [0]
        for point, departure in enumerate(DEPARTURES):
            oldoff, oldnodes, oldarr, oldcost = routed['G0'][point]
            off, nodes, arrival, costs = routed['G1'][point]
            row = np.repeat(np.arange(len(gids)), np.diff(off))
            keys = row*4096+self.cells[nodes]
            unique, back = np.unique(keys[::-1], return_index=True); last = len(keys)-1-back
            counts = np.bincount(unique//4096, minlength=len(gids))
            imageoff.extend((np.cumsum(counts)+imageoff[-1]).tolist())
            image_cells.append((unique % 4096).astype(np.uint16))
            image_progress.append((2*(arrival[last]-departure)/costs[unique//4096]-1).astype(np.float32))
            for i in range(len(gids)):
                sl, oldsl = slice(off[i], off[i+1]), slice(oldoff[i], oldoff[i+1])
                evidence, internal = point_evidence(self.native['G0'], self.native['G1'], self.cells,
                    self.hotmask, self.shifted, oldnodes[oldsl], oldcost[i], nodes[sl], arrival[sl], costs[i])
                flags.append(evidence); hotcells.append(internal); hotoff.append(hotoff[-1]+len(internal))
        temporary = target.with_suffix('.tmp.h5')
        if temporary.exists():
            archive = self.run/'failures'/self.city/block
            archive.mkdir(parents=True, exist_ok=True)
            temporary.replace(archive/(temporary.name+'.'+self.attempt))
        with h5py.File(temporary, 'w') as f:
            f.create_dataset('group_ids', data=gids)
            for stage in ('G0', 'G1'):
                offsets = [0]; node_parts, arrival_parts, cost_parts = [], [], []
                for off, nodes, arrival, costs in routed[stage]:
                    offsets.extend((off[1:]+offsets[-1]).tolist())
                    node_parts.append(nodes); arrival_parts.append(arrival); cost_parts.append(costs)
                group = f.create_group(stage)
                for name, value in dict(path_offsets=np.asarray(offsets, np.int64),
                    node_indices=np.concatenate(node_parts), arrival_times=np.concatenate(arrival_parts),
                    cost_seconds=np.concatenate(cost_parts)).items():
                    group.create_dataset(name, data=value, compression='lzf' if value.size else None)
            for name, value in dict(cell_offsets=np.asarray(imageoff, np.int64),
                cells=np.concatenate(image_cells), progress=np.concatenate(image_progress),
                point_flags=np.asarray(flags, bool), internal_hot_offsets=np.asarray(hotoff, np.int64),
                internal_hot_cells=np.concatenate(hotcells)).items():
                f.create_dataset(name, data=value, compression='lzf' if value.size else None)
            f.attrs.update(status='COMPLETE', city=self.city, source=int(source), groups=len(gids),
                points_per_group=2, grid_dim=64, departure_order=json.dumps(DEPARTURES),
                row_order='departure-major; group_ids sorted within source',
                group_id_semantics='source_index*n_nodes+destination_index',
                flag_names=json.dumps(FLAG_NAMES), sssp_calls=4, point_paths_in_this_successful_bundle=4*len(gids),
                edge_replay_max_error=max_error, rational_witnesses=json.dumps(witnesses),
                elapsed_seconds=time.monotonic()-started, completed_unix=time.time())
        temporary.replace(target)
        return str(target)

    def make_block(self, block, gids):
        assert block.isidentifier()
        gids = np.sort(np.asarray(gids, np.int64))
        assert len(gids) and len(np.unique(gids)) == len(gids)
        directory = self.cache/block; directory.mkdir(parents=True, exist_ok=True)
        request = directory/'requested_gids.npy'
        if request.exists(): assert np.array_equal(np.load(request), gids)
        else: atomic_array(request, gids)
        missing = np.asarray([gid for gid in gids if int(gid) not in self.index], np.int64)
        source, start, count = np.unique(missing//self.n, return_index=True, return_counts=True)
        completed = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            jobs = [pool.submit(self.source_job, block, int(s), missing[i:i+k])
                    for s, i, k in zip(source, start, count)]
            try:
                for job in as_completed(jobs):
                    self.register(Path(job.result())); completed += 1
                    if completed % 32 == 0 or completed == len(jobs):
                        progress = dict(status='GENERATING_GT', city=self.city, block=block,
                            source_files_completed=completed, source_files_required=len(jobs),
                            requested_groups=len(gids), missing_groups_at_start=len(missing),
                            cached_groups_total=len(self.index), elapsed_seconds=time.monotonic()-self.started)
                        atomic_json(self.folder/'GT_progress.json', progress)
                        print(json.dumps(progress), flush=True)
            except BaseException:
                self.stop.set(); raise
        assert all(int(gid) in self.index for gid in gids)
        summary = dict(status='COMPLETE', city=self.city, block=block, requested_groups=len(gids),
            new_successful_groups=len(missing), reused_groups=len(gids)-len(missing),
            completed_source_files=len(jobs), no_GT_queries_for_reused_groups=True)
        atomic_json(directory/'block_summary.json', summary)
        self.check_inputs()
        return summary

    def flags(self, gids):
        result = np.empty((len(gids), 2, len(FLAG_NAMES)), bool); work = {}
        for out, gid in enumerate(gids):
            file, row = self.index[int(gid)]; work.setdefault(file, []).append((out, row))
        for file, rows in work.items():
            with h5py.File(file, 'r') as f:
                count = len(f['group_ids']); flags = f['point_flags'][:]
                for out, row in rows: result[out] = flags[[row, count+row]]
        return result

    def save_index(self):
        gids = np.asarray(sorted(self.index), np.int64)
        paths = np.asarray([str(Path(self.index[int(gid)][0]).relative_to(self.run)) for gid in gids])
        rows = np.asarray([self.index[int(gid)][1] for gid in gids], np.int64)
        temporary = self.folder/'cache_index.tmp.npz'
        np.savez(temporary, group_ids=gids, paths=paths, rows=rows)
        temporary.replace(self.folder/'cache_index.npz')

    def close(self):
        self.events.close()
        for native in self.native.values(): native.lib.e1_free(native.handle)
        fcntl.flock(self.lock, fcntl.LOCK_UN); self.lock.close()


def accounting(run, city):
    intents, complete = {}, {}
    for path in (Path(run)/'ledger'/city).glob('calls_*.jsonl'):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e['event'] == 'CALL_INTENT':
                assert e['call_id'] not in intents; intents[e['call_id']] = e
            elif e['event'] == 'CALL_COMPLETE':
                assert e['call_id'] not in complete; complete[e['call_id']] = e
    assert set(complete) <= set(intents)
    for key in complete: assert complete[key]['completed_point_paths'] == intents[key]['requested_point_paths']
    unresolved = [intents[key] for key in intents.keys()-complete.keys()]
    return dict(requested_point_path_restorations=sum(e['requested_point_paths'] for e in intents.values()),
        known_completed_point_path_restorations=sum(e['completed_point_paths'] for e in complete.values()),
        requested_but_completion_unconfirmed=sum(e['requested_point_paths'] for e in unresolved),
        requested_sssp_calls=len(intents), completed_sssp_calls=len(complete), unresolved_calls=unresolved,
        query_wall_seconds_sum=sum(e['elapsed_seconds'] for e in complete.values()),
        query_thread_cpu_seconds_sum=sum(e['thread_cpu_seconds'] for e in complete.values()),
        sum_wall_scope='sum of possibly parallel query durations, not end-to-end wall time')
