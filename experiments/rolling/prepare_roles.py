"""Freeze cadence-study roles without reading any model outputs or new GT."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from common import ROOT, PARENT, OLD, atomic, stamp, frozen_npz
sys.path.insert(0, str(PARENT / 'e1_v1/code'))
from p0_plan import role_codes
from p0_inventory import BBOX, cell_xy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', choices=['BJ', 'XA'], required=True)
    args = parser.parse_args()
    city = args.city
    folder = ROOT / 'inputs/rolling_roles'
    folder.mkdir(parents=True, exist_ok=True)
    reportfile = folder / f'{city}.json'
    if reportfile.exists():
        old = json.loads(reportfile.read_text())
        assert old['status'] == 'FROZEN_ROLES_NO_NEW_GT_OR_MODEL_CALLS'
        assert stamp(folder / f'{city}.npz') == old['cohort']
        print(json.dumps(old), flush=True)
        return
    e1 = PARENT / 'e1_v1/data' / city
    coords = np.load(e1 / 'coordinates.npy')
    cells = np.load(e1 / 'cell64.npy')
    n = len(coords)
    payload = {}
    sources = []
    fit = []
    cal = []
    for hour in range(1, 25):
        trainpath = OLD / f'rolling_data_diverse/{city}/v{hour:02d}/sdc_Afit.npz'
        calpath = OLD / f'rolling_data/{city}/v{hour:02d}/Acal.npz'
        with np.load(trainpath) as data:
            train = data['group_ids']
        with np.load(calpath) as data:
            calibration = data['group_ids']
        assert train.shape == (256,) and calibration.shape == (64,)
        assert np.all(role_codes(train // n, train % n, n) == 0)
        assert np.all(role_codes(calibration // n, calibration % n, n) == 2)
        fit.extend([train] * 4)
        cal.extend([calibration] * 4)
        sources.append(dict(hour=hour, train=stamp(trainpath), cal=stamp(calpath)))
    payload['train_group_ids'] = np.asarray(fit, dtype=np.int64)
    payload['cal_group_ids'] = np.asarray(cal, dtype=np.int64)
    devpath = OLD / f'inputs/{city}/rolling_test_cohort.npz'
    with np.load(devpath) as data:
        dev = data['group_ids'].copy()
    assert dev.shape == (128,)
    assert np.all(role_codes(dev // n, dev % n, n) == 4)
    payload['dev_group_ids'] = dev
    used = set()
    exclusions = []

    def add_gids(values, path, key):
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        a, b = values // n, values % n
        assert np.all((a >= 0) & (a < n) & (b >= 0) & (b < n)), (str(path), key)
        used.update(map(int, values))
        used.update(map(int, b * n + a))
        exclusions.append(dict(source=stamp(path), key=key, identities=len(values), reverse_excluded=True))

    groupspath = e1 / 'groups.npy'
    groups = np.load(groupspath)
    add_gids(groups[:, 0] * n + groups[:, 1], groupspath, 'all_G0_roles')
    # All original G1 adaptation/select/cal/test and preselected candidate pools.
    for path in sorted((PARENT / 'e2_v1/data' / city).glob('*gids.npy')):
        add_gids(np.load(path), path, 'all_G1_roles_or_candidate_gids')
    paths = [PARENT / f'e4_full_v2/inputs/{city}/{snap}/queries.npz' for snap in ['G0', 'G1']]
    paths += list((OLD / f'inputs/{city}').glob('*.npz'))
    paths += list((OLD / f'inputs/confirmation_interval/{city}').glob('*.npz'))
    paths += [OLD / f'warmup_data/{city}/confirmation_cohort.npz']
    paths += [OLD / f'warmup_data/{city}/roles.npz']
    for path in paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if key == 'group_ids' or key.endswith('_group_ids'):
                    add_gids(data[key], path, key)
    # Our inherited train/cal/dev are also explicitly covered, including reverse.
    for key, values in payload.items():
        used.update(map(int, values.reshape(-1)))
        used.update(map(int, (values % n * n + values // n).reshape(-1)))
    # Historical exposure metadata uses original node IDs, not coordinate indices.
    capacity = PARENT / f'e1_v1/p0/{city}_sampler_capacity.json'
    for info in json.loads(capacity.read_text())['exposure_files']:
        path = Path(info['path'])
        records = json.loads(path.read_text())
        records = records.get('queries', records.get('intervals', [])) if isinstance(records, dict) else records
        values = []
        for query in records:
            if 'source' not in query or 'destination' not in query:
                continue
            ids = np.asarray([query['source'], query['destination']], dtype=np.int64)
            pair = np.searchsorted(coords[:, 0], ids)
            if np.all(pair < n) and np.array_equal(coords[pair, 0], ids):
                values.append(int(pair[0]) * n + int(pair[1]))
        add_gids(values, path, 'historical_node_ID_exposure')
    xy = cell_xy(coords[:, 1:], BBOX[city], 32)
    graph = np.load(e1 / 'routing_graph.npz')
    _, components = connected_components(csr_matrix((np.ones(len(graph['u']), np.uint8),
        (graph['u'], graph['v'])), shape=(n, n)), directed=True, connection='strong')
    rng = np.random.default_rng(np.random.SeedSequence([20261031, city == 'BJ', 900, 4]))
    selected = []
    draws = 0
    while len(selected) < 64:
        a, b = rng.integers(0, n, size=(2, 32768))
        distance = np.abs(xy[a] - xy[b]).sum(1)
        keep = (distance >= 49) & (distance <= 56) & (components[a] == components[b]) & (role_codes(a, b, n) == 4)
        draws += len(a)
        for s, t in zip(a[keep], b[keep]):
            gid, reverse = int(s) * n + int(t), int(t) * n + int(s)
            if gid in used or reverse in used:
                continue
            selected.append(gid)
            used.add(gid)
            used.add(reverse)
            if len(selected) == 64:
                break
    final = np.asarray(selected, dtype=np.int64)
    payload['final_group_ids'] = final
    payload['departures'] = np.arange(0, 86400, 900, dtype=np.int64)
    payload['slots'] = np.arange(1, 97, dtype=np.int16)
    for role in ['train', 'cal', 'dev', 'final']:
        gids = payload[f'{role}_group_ids']
        pair = np.stack((gids // n, gids % n), axis=-1)
        payload[f'{role}_node_indices'] = pair
        payload[f'{role}_node_pairs'] = coords[pair, 0]
        payload[f'{role}_endpoints'] = cells[pair]
        if role in ['dev', 'final']:
            payload[f'{role}_distance32'] = np.abs(xy[pair[:, 0]] - xy[pair[:, 1]]).sum(1)
    path = folder / f'{city}.npz'
    frozen_npz(path, **payload)
    exclusionpath = folder / f'{city}_exclusions.json'
    atomic(exclusionpath, dict(status='OUTCOME_BLIND_IDENTITY_EXCLUSION_LOG', sources=exclusions,
        draws=draws, all_known_used_including_reverse=len(used), prediction_metrics_read=0, new_GT_read=0))
    report = dict(status='FROZEN_ROLES_NO_NEW_GT_OR_MODEL_CALLS', city=city, cohort=stamp(path),
        slots=96, train_points_per_slot=256, calibration_points_per_slot=64, development_ODs=128,
        final_ODs=64, final_points=6144, inherited_hourly_fit_sources=sources,
        inherited_80_20_identity_only=True, new_slot_target_qualification_performed=False,
        scope='offline cadence replay; no causal live prediction claim',
        evaluation_departure='(slot-1)*900', graph_release='slot G1 knots now available before FT and evaluation',
        final_selection='uniform seeded admissible role4 ODs before any new GT or model calls',
        exclusion_log=stamp(exclusionpath), development_source=stamp(devpath), hash_verification=False,
        created_unix=time.time())
    atomic(reportfile, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
