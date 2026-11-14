"""Native 64-cell-grid change detection and update qualification."""
import numpy as np

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


