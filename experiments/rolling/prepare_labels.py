"""One-time route labels for shared quarter-hour roles, with exact old-input reuse."""
import argparse
import fcntl
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from common import ROOT, PARENT, OLD, BASE, atomic, stamp, frozen_npz, frozen_json
sys.path.insert(0, str(PARENT / 'e1_v1/code'))
from point_data import Native
from p0_inventory import BBOX


def snapshot_for(city, slot, graph, coords):
    if slot % 4 == 0:
        path = OLD / f'graphs/{city}/v{slot // 4:02d}_snapshot.txt'
        assert path.exists()
        return path
    path = ROOT / f'graphs/{city}/v{slot:03d}_snapshot.txt'
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            tmp = path.with_suffix('.tmp.txt')
            with tmp.open('w') as stream:
                stream.write(f'{len(coords)} {len(graph["u"])}\n')
                np.savetxt(stream, np.column_stack((coords[graph['u'], 0],
                    coords[graph['v'], 0], graph['knots'])), fmt='%d')
            tmp.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['BJ', 'XA'], required=True)
    parser.add_argument('--part', choices=[0, 1], type=int, required=True)
    args = parser.parse_args()
    city, part = args.city, args.part
    assert len(os.sched_getaffinity(0)) == 1
    rolepath = ROOT / f'inputs/rolling_roles/{city}.npz'
    roles = dict(np.load(rolepath))
    frozen_role_stamp = stamp(rolepath)
    coords = np.load(PARENT / f'e1_v1/data/{city}/coordinates.npy')
    cells = np.load(PARENT / f'e1_v1/data/{city}/cell64.npy')
    n = len(coords)
    inventory = json.loads((PARENT / f'e1_v1/p0/{city}_G0_inventory.json').read_text())
    box = BBOX[city]
    low = np.asarray([box[0], box[2]])
    span = np.asarray([box[1] - box[0], box[3] - box[2]])
    out = Path(os.environ['E4_OUTPUT_DIR'])
    assigned = list(range(1 + part, 97, 2))
    new_calls = reused_points = new_points = completed = 0
    started = time.monotonic()

    def progress(status, **extra):
        atomic(out / 'progress.json', dict(status=status, city=city, part=part,
            assigned_slots=assigned, completed_slots=completed, required_slots=48,
            new_solver_calls=new_calls, new_route_points=new_points,
            reused_route_points=reused_points, pid=os.getpid(), host=socket.gethostname(),
            model_calls=0, updated_unix=time.time(), elapsed_seconds=time.monotonic() - started, **extra))

    for slot in assigned:
        folder = ROOT / f'data/rolling/{city}/v{slot:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'worker.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (folder / 'prepared.json').exists():
                previous = json.loads((folder / 'prepared.json').read_text())
                assert previous['status'] == 'ALL_SHARED_SLOT_LABELS_READY_FOR_INDEPENDENT_REVIEW'
                assert previous['cohort'] == frozen_role_stamp
                completed += 1
                progress('REUSING_COMPLETE_SLOT', slot=slot)
                continue
            assert stamp(rolepath) == frozen_role_stamp
            graphpath = ROOT / f'graphs/{city}/v{slot:03d}.npz'
            graph_record = json.loads(graphpath.with_suffix('.json').read_text())
            graph = dict(np.load(graphpath))
            graph_stamp = stamp(graphpath)
            assert graph_record['graph'] == graph_stamp
            departure = (slot - 1) * 900
            snapshot = snapshot_for(city, slot, graph, coords)
            native = None
            selected = {role: (roles[f'{role}_group_ids'][slot - 1] if role in ['train', 'cal']
                else roles[f'{role}_group_ids']) for role in ['train', 'cal', 'dev', 'final']}
            allgids = np.unique(np.concatenate(list(selected.values())))
            assert len(allgids) == 512
            routes = {}
            lineages = {}
            try:
                # Every fourth graph is identical to the accepted legacy hourly graph.
                # Only its fourth-departure raw route can therefore be reused here.
                if slot % 4 == 0:
                    hour = slot // 4
                    for gid in allgids:
                        candidates = [OLD / f'rolling_data/{city}/v{hour:02d}/cache/new/{int(gid)}_3.npz',
                            OLD / f'rolling_data_diverse/{city}/v{hour:02d}/cache/{int(gid)}_3.npz']
                        for path in candidates:
                            if not path.exists():
                                continue
                            with np.load(path) as data:
                                assert int(data['group_id']) == int(gid)
                                assert int(data['departure']) == departure
                                nodes, arrivals = data['nodes'], data['arrivals']
                                cost = float(data['costs'][0])
                            assert nodes[0] == gid // n and nodes[-1] == gid % n
                            audit = path.with_suffix('.audit.json')
                            assert json.loads(audit.read_text())['status'] == 'PASSED'
                            routes[int(gid)] = (nodes, arrivals, cost)
                            lineages[int(gid)] = dict(kind='EXACT_LEGACY_GRAPH_OD_DEPARTURE_REUSE',
                                raw=stamp(path), audit=stamp(audit), prior_graph=graph_record['exact_legacy_graph_reuse'])
                            reused_points += 1
                            break
                missing = np.asarray([gid for gid in allgids if int(gid) not in routes], dtype=np.int64)
                if len(missing):
                    native = Native(BASE, city, snapshot=snapshot, coordinates=inventory['paths']['coordinates'])
                    assert np.array_equal(native.u, graph['u']) and np.array_equal(native.v, graph['v'])
                    assert np.array_equal(native.knots, graph['knots'])
                for source in np.unique(missing // n):
                    gids = missing[missing // n == source]
                    targets = (gids % n).astype(np.int32)
                    path = folder / f'cache/s{int(source):07d}.npz'
                    path.parent.mkdir(exist_ok=True)
                    intentpath = path.with_suffix('.intent.json')
                    identity = dict(city=city, slot=slot, departure=departure,
                        source=int(source), group_ids=gids.tolist(), graph=graph_stamp)
                    if path.exists():
                        assert json.loads(intentpath.read_text())['identity'] == identity
                        with np.load(path) as raw:
                            assert np.array_equal(raw['group_ids'], gids)
                            offsets, nodes = raw['offsets'], raw['nodes']
                            arrivals, costs = raw['arrivals'], raw['costs']
                    else:
                        assert not intentpath.exists(), 'Unresolved solver intent: no automatic repeated call'
                        frozen_json(intentpath, dict(status='CALL_INTENT', identity=identity,
                            pid=os.getpid(), host=socket.gethostname(), started_unix=time.time()))
                        timer = time.monotonic()
                        offsets, nodes, arrivals, costs = native.query(int(source), departure, targets)
                        seconds = time.monotonic() - timer
                        new_calls += 1
                        new_points += len(gids)
                        # Raw result commits before any audit or transformation.
                        frozen_npz(path, group_ids=gids, departure=departure, source=int(source),
                            targets=targets, offsets=offsets, nodes=nodes, arrivals=arrivals,
                            costs=costs, query_seconds=seconds)
                    errors = native.replay(int(source), departure, targets, offsets, nodes, arrivals, costs)
                    exact = []
                    for index, gid in enumerate(gids):
                        sl = slice(offsets[index], offsets[index + 1])
                        path_nodes, path_arrivals = nodes[sl], arrivals[sl]
                        assert path_nodes[0] == source and path_nodes[-1] == gid % n
                        assert path_arrivals[0] == departure and costs[index] > 0
                        rational = native.rational_replay(path_nodes, departure)
                        assert abs(float(rational) - path_arrivals[-1]) < 1e-7
                        exact.append(str(rational))
                        routes[int(gid)] = (path_nodes.copy(), path_arrivals.copy(), float(costs[index]))
                        lineages[int(gid)] = dict(kind='NEW_EXACT_GRAPH_OD_DEPARTURE', raw=stamp(path),
                            route_index=index, intent=stamp(intentpath), audit=str(path.with_suffix('.audit.json')))
                    frozen_json(path.with_suffix('.audit.json'), dict(status='EXECUTION_REPLAY_PASSED_PENDING_EXTERNAL_AUDIT',
                        identity=identity, raw=stamp(path), original_arc_replay=True,
                        exact_rational_arrivals=exact, max_float_replay_error=float(errors)))
                    progress('GENERATING_SLOT', slot=slot, current_routes=len(routes), required_routes=512)
                assert len(routes) == 512
                role_reports = {}
                for role, gids in selected.items():
                    pairs = np.column_stack((gids // n, gids % n))
                    image = np.full((len(gids), 2, 4096), -1., np.float32)
                    odt = np.empty((len(gids), 5), np.float32)
                    odt[:, :2] = 2 * (coords[pairs[:, 0], 1:] - low) / span - 1
                    odt[:, 2:4] = 2 * (coords[pairs[:, 1], 1:] - low) / span - 1
                    odt[:, 4] = 2 * departure / 86400 - 1
                    nodepieces, arrpieces, costs = [], [], []
                    for index, gid in enumerate(gids):
                        nodes, arrivals, cost = routes[int(gid)]
                        unique, reverse = np.unique(cells[nodes][::-1], return_index=True)
                        last = len(nodes) - 1 - reverse
                        image[index, 0, unique] = 1.
                        image[index, 1, unique] = (2 * (arrivals[last] - departure) / cost - 1).astype(np.float32)
                        nodepieces.append(nodes)
                        arrpieces.append(arrivals)
                        costs.append(cost)
                    rolefile = folder / f'{role}.npz'
                    frozen_npz(rolefile, image=image.reshape(-1, 2, 64, 64), odt=odt,
                        query_ids=gids * 96 + slot - 1, group_ids=gids,
                        node_indices=pairs, node_pairs=coords[pairs, 0], endpoints=cells[pairs],
                        departure=np.int64(departure), graph_slot=np.int64(slot),
                        route_offsets=np.r_[0, np.cumsum([len(x) for x in nodepieces])].astype(np.int64),
                        route_nodes=np.concatenate(nodepieces), route_arrivals=np.concatenate(arrpieces),
                        cost_seconds=np.asarray(costs, dtype=np.float64))
                    lineagefile = folder / f'{role}_lineage.json'
                    frozen_json(lineagefile, [dict(group_id=int(g), **lineages[int(g)]) for g in gids])
                    role_reports[role] = dict(points=len(gids), data=stamp(rolefile), lineage=stamp(lineagefile))
                assert stamp(rolepath) == frozen_role_stamp and stamp(graphpath) == graph_stamp
                atomic(folder / 'prepared.json', dict(status='ALL_SHARED_SLOT_LABELS_READY_FOR_INDEPENDENT_REVIEW',
                    city=city, slot=slot, release_available_clock_seconds=departure,
                    evaluate_departure_seconds=departure, cohort=frozen_role_stamp, graph=graph_stamp,
                    snapshot=stamp(snapshot), roles=role_reports, points=512,
                    reused_points=sum(v['kind'].startswith('EXACT_LEGACY') for v in lineages.values()),
                    new_points=sum(v['kind'].startswith('NEW_') for v in lineages.values()),
                    all_original_arcs_replayed=True, all_new_routes_rational_replayed=True,
                    model_calls=0, hash_verification=False))
                completed += 1
                progress('SLOT_COMPLETE', slot=slot)
            finally:
                if native is not None:
                    native.lib.e1_free(native.handle)
                    native.handle = None
    progress('FINISHED')


if __name__ == '__main__':
    main()
