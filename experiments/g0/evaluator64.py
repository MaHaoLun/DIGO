"""Frozen SSP section 8–9 occupancy evaluation; no test threshold selection.

NumPy metric helpers do not import torch. Inference imports the sibling trainer
lazily and uses independent PCG64 streams for each public query identity.
"""
from __future__ import annotations

import csv
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

EMPTY_THRESHOLD = float(np.nextafter(np.float32(1), np.float32(np.inf)))
TARGET = 0.88
TIE_TOLERANCE = 1e-8
EXACT_SWEEP_BLOCK_SIZE = 1_000_000
COARSE_THRESHOLDS = np.linspace(0.0, 1.0, 101, dtype=np.float64)
_MASK64 = (1 << 64) - 1


class NumericalEvaluationError(ValueError):
    """Evaluation failed before clipping; an audit is persisted by the caller."""


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)


def _mix64(value):
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def identity_seed(seed, query_id, city, snapshot_id, role):
    """Integer identity mixing, never a data/file content digest.

    Length-delimited UTF-8 fields prevent ambiguous city/snapshot/role joins.
    Checkpoint, training seed/RNG, order and batch size are deliberately absent.
    """
    state = _mix64(seed)
    for field in (city, snapshot_id, role):
        encoded = str(field).encode('utf-8')
        state = _mix64(state ^ len(encoded))
        for byte in encoded:
            state = _mix64(state ^ byte)
    return _mix64(state ^ int(query_id))


def fixed_query_noise(query_ids, *, eval_seed, city, snapshot_id, role,
                      shape=(2, 64, 64)):
    noise = np.empty((len(query_ids), *shape), dtype=np.float32)
    for index, query_id in enumerate(query_ids):
        rng = np.random.Generator(np.random.PCG64(
            identity_seed(eval_seed, query_id, city, snapshot_id, role)))
        noise[index] = rng.standard_normal(shape, dtype=np.float32)
    return noise


def _ratios(tp, predicted, positives):
    """All counters enter as int64; every ratio is computed in float64."""
    tp = np.asarray(tp, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    positives = np.asarray(positives, dtype=np.int64)
    tp, predicted, positives = np.broadcast_arrays(tp, predicted, positives)
    recall = np.divide(tp, positives, out=np.zeros(tp.shape, np.float64),
                       where=positives != 0)
    precision = np.divide(tp, predicted, out=np.zeros(tp.shape, np.float64),
                          where=predicted != 0)
    denominator = predicted + positives
    f1 = np.divide(2.0 * tp, denominator,
                   out=np.zeros(tp.shape, np.float64), where=denominator != 0)
    return recall, precision, f1


def _point(tp, predicted, positives, rows, threshold):
    tp, predicted, positives = int(tp), int(predicted), int(positives)
    recall, precision, f1 = _ratios(tp, predicted, positives)
    return dict(threshold=float(threshold), q=float(min(recall, f1)),
                recall_micro=float(recall), precision_micro=float(precision),
                f1_micro=float(f1), tp=tp, fp=predicted-tp, fn=positives-tp,
                predicted_cells=predicted, gt_cells=positives,
                mean_predicted_cells=float(predicted / rows),
                mean_fp_cells=float((predicted-tp) / rows), rows=int(rows),
                meets_target=bool(recall >= TARGET and f1 >= TARGET), nfe=1)


def _validate_arrays(scores, gt, endpoints, road_support):
    scores = np.asarray(scores)
    gt = np.asarray(gt)
    if scores.dtype != np.float32:
        raise ValueError('score contract requires float32; do not silently merge higher-precision values')
    if scores.ndim == 3:
        scores = scores.reshape(len(scores), -1)
    if gt.ndim == 3:
        gt = gt.reshape(len(gt), -1)
    if scores.ndim != 2 or scores.shape != gt.shape or len(scores) == 0:
        raise ValueError('scores and GT must share a nonempty [N,cells] shape')
    if gt.dtype != np.bool_:
        raise ValueError('GT must be boolean occupancy, without support filtering')
    road_support = np.asarray(road_support, dtype=bool)
    endpoints = np.asarray(endpoints)
    if road_support.shape != (scores.shape[1],):
        raise ValueError('road_support must be static [cells]')
    if endpoints.shape != (len(scores), 2) or endpoints.dtype.kind not in 'iu':
        raise ValueError('known endpoints must have integer [N,2] shape')
    if np.any(endpoints < 0) or np.any(endpoints >= scores.shape[1]):
        raise ValueError('known endpoint outside fixed grid')
    # Chunked validation avoids a full-size transient boolean allocation.
    for begin in range(0, len(scores), 1024):
        block = scores[begin:begin+1024]
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise NumericalEvaluationError('clipped-affine scores must be finite in [0,1]')
    return scores, gt, endpoints.astype(np.int64, copy=False), road_support


def _endpoint_mask(endpoints, cells):
    mask = np.zeros((len(endpoints), cells), dtype=bool)
    mask[np.arange(len(endpoints))[:, None], endpoints] = True
    return mask


def _histogram(values, truth):
    # bucket k contains values satisfying exactly k of the 101 >= thresholds.
    bins = np.searchsorted(COARSE_THRESHOLDS, values, side='right')
    predicted = np.bincount(bins, minlength=102).astype(np.int64, copy=False)
    true_positive = np.bincount(bins[truth], minlength=102).astype(np.int64, copy=False)
    return predicted, true_positive


def exact_threshold_sweep(scores, gt, endpoints, road_support):
    """Same-score atomic scan with O(E log E) sort and O(E) cumulative counts.

    E contains supported cells excluding mandatory endpoints. Other score
    thresholds cannot change handoff counts, and are dominated in the higher-tau
    tie break by a retained threshold (including EMPTY_THRESHOLD). Thus skipping
    those no-op events has exactly the all-distinct-score optimum and tie winner.
    The raw and handoff coarse reporting curves still cover all 101 thresholds.
    """
    scores, gt, endpoints, road_support = _validate_arrays(
        scores, gt, endpoints, road_support)
    rows, cells = scores.shape
    duplicate_endpoints = endpoints[:, 0] == endpoints[:, 1]
    endpoint_predicted = int(2*rows - duplicate_endpoints.sum(dtype=np.int64))
    endpoint_tp = int(gt[np.arange(rows), endpoints[:, 0]].sum(dtype=np.int64)
                      + gt[np.arange(rows)[~duplicate_endpoints],
                           endpoints[~duplicate_endpoints, 1]].sum(dtype=np.int64))
    supported_endpoints = (road_support[endpoints[:, 0]].sum(dtype=np.int64)
                           + road_support[endpoints[~duplicate_endpoints, 1]].sum(dtype=np.int64))
    event_count = int(rows * road_support.sum(dtype=np.int64) - supported_endpoints)
    event_scores = np.empty(event_count, dtype=np.float32)
    event_gt = np.empty(event_count, dtype=bool)
    raw_hist = np.zeros(102, dtype=np.int64)
    raw_tp_hist = np.zeros(102, dtype=np.int64)
    hand_hist = np.zeros(102, dtype=np.int64)
    hand_tp_hist = np.zeros(102, dtype=np.int64)
    gt_count = 0
    cursor = 0
    for begin in range(0, rows, 1024):
        block, truth = scores[begin:begin+1024], gt[begin:begin+1024]
        endpoint = _endpoint_mask(endpoints[begin:begin+1024], cells)
        eligible = road_support[None, :] & ~endpoint
        count = int(eligible.sum(dtype=np.int64))
        values, labels = block[eligible], truth[eligible]
        event_scores[cursor:cursor+count] = values
        event_gt[cursor:cursor+count] = labels
        cursor += count
        hist, tp_hist = _histogram(block.ravel(), truth.ravel())
        raw_hist += hist
        raw_tp_hist += tp_hist
        hist, tp_hist = _histogram(values, labels)
        hand_hist += hist
        hand_tp_hist += tp_hist
        gt_count += int(truth.sum(dtype=np.int64))
    if cursor != event_count:
        raise AssertionError('endpoint accounting mismatch')
    raw_pred = np.cumsum(raw_hist[:0:-1], dtype=np.int64)[::-1]
    raw_tp = np.cumsum(raw_tp_hist[:0:-1], dtype=np.int64)[::-1]
    hand_pred = np.cumsum(hand_hist[:0:-1], dtype=np.int64)[::-1] + endpoint_predicted
    hand_tp = np.cumsum(hand_tp_hist[:0:-1], dtype=np.int64)[::-1] + endpoint_tp
    coarse = []
    for index, threshold in enumerate(COARSE_THRESHOLDS):
        coarse.append(dict(threshold=float(threshold),
                           raw=_point(raw_tp[index], raw_pred[index], gt_count, rows, threshold),
                           handoff=_point(hand_tp[index], hand_pred[index], gt_count, rows, threshold)))
    empty = _point(endpoint_tp, endpoint_predicted, gt_count, rows, EMPTY_THRESHOLD)
    best = dict(empty)
    strict_best = dict(empty)
    exact_max_q = best['q']
    max_f1_at_target_recall = best['f1_micro'] if best['recall_micro'] >= TARGET else None
    unique_effective = 0
    if event_count:
        order = np.argsort(event_scores, kind='stable')
        sorted_scores = event_scores[order][::-1]
        sorted_gt = event_gt[order][::-1]
        del order, event_scores, event_gt
        seen_tp = 0
        block_size = EXACT_SWEEP_BLOCK_SIZE
        for begin in range(0, event_count, block_size):
            end = min(begin + block_size, event_count)
            values = sorted_scores[begin:end]
            running_tp = np.cumsum(sorted_gt[begin:end], dtype=np.int64)
            # A same-score group may straddle blocks: evaluate only its last row.
            boundary = np.empty(end-begin, dtype=bool)
            boundary[:-1] = values[:-1] != values[1:]
            boundary[-1] = end == event_count or values[-1] != sorted_scores[end]
            indices = np.flatnonzero(boundary)
            tp = endpoint_tp + seen_tp + running_tp[indices]
            predicted = endpoint_predicted + begin + indices.astype(np.int64) + 1
            recall, precision, f1 = _ratios(tp, predicted, gt_count)
            q = np.minimum(recall, f1)
            if len(q):
                unique_effective += len(q)
                strict_index = int(np.argmax(q))
                if float(q[strict_index]) > exact_max_q:
                    exact_max_q = float(q[strict_index])
                    strict_best = _point(tp[strict_index], predicted[strict_index], gt_count,
                                         rows, values[indices[strict_index]])
                feasible = f1[recall >= TARGET]
                if len(feasible):
                    maximum = float(feasible.max())
                    max_f1_at_target_recall = maximum if max_f1_at_target_recall is None else max(maximum, max_f1_at_target_recall)
            seen_tp += int(running_tp[-1])
        # Second pass uses the GLOBAL optimum's tolerance band. Keeping only a
        # local-band winner in the first pass can lose an earlier valid candidate
        # when a later score group raises the maximum. Region grows monotonically,
        # so the first qualifying point is the deterministic tie winner.
        if empty['q'] < exact_max_q-TIE_TOLERANCE:
            seen_tp = 0
            for begin in range(0, event_count, block_size):
                end = min(begin+block_size, event_count)
                values = sorted_scores[begin:end]
                running_tp = np.cumsum(sorted_gt[begin:end], dtype=np.int64)
                boundary = np.empty(end-begin, dtype=bool)
                boundary[:-1] = values[:-1] != values[1:]
                boundary[-1] = end == event_count or values[-1] != sorted_scores[end]
                indices = np.flatnonzero(boundary)
                tp = endpoint_tp+seen_tp+running_tp[indices]
                predicted = endpoint_predicted+begin+indices.astype(np.int64)+1
                recall, _, f1 = _ratios(tp, predicted, gt_count)
                qualifying = np.flatnonzero(np.minimum(recall, f1) >= exact_max_q-TIE_TOLERANCE)
                if len(qualifying):
                    selected = int(qualifying[0])
                    best = _point(tp[selected], predicted[selected], gt_count,
                                  rows, values[indices[selected]])
                    break
                seen_tp += int(running_tp[-1])
    return dict(best=best, max_q=exact_max_q, max_q_point=strict_best, coarse_curve=coarse,
                empty_core_boundary=empty,
                exact_scan=dict(max_q=exact_max_q,
                                effective_unique_score_groups=int(unique_effective),
                                eligible_cell_events=event_count,
                                all_score_thresholds_equivalent=True,
                                max_f1_at_recall_ge_088=max_f1_at_target_recall))


def _count_rows(predicted, gt):
    tp = np.count_nonzero(predicted & gt, axis=1).astype(np.int64)
    npred = np.count_nonzero(predicted, axis=1).astype(np.int64)
    ngt = np.count_nonzero(gt, axis=1).astype(np.int64)
    return np.column_stack((tp, npred-tp, ngt-tp)).astype(np.int64, copy=False)


def _micro(counts):
    tp, fp, fn = np.asarray(counts, dtype=np.int64).sum(axis=0, dtype=np.int64)
    recall, precision, f1 = _ratios(tp, tp+fp, tp+fn)
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), recall_micro=float(recall),
                precision_micro=float(precision), f1_micro=float(f1),
                q=float(min(recall, f1)))


def _macro(counts, mask=None):
    if mask is not None:
        counts = counts[mask]
    if len(counts) == 0:
        return dict(units=0, recall=None, precision=None, f1=None)
    tp, fp, fn = counts.T
    recall, precision, f1 = _ratios(tp, tp+fp, tp+fn)
    return dict(units=len(counts), recall=float(recall.mean()),
                precision=float(precision.mean()), f1=float(f1.mean()))


def _group_counts(counts, od_groups):
    groups, inverse = np.unique(od_groups, return_inverse=True)
    grouped = np.zeros((len(groups), 3), dtype=np.int64)
    np.add.at(grouped, inverse, counts)
    return groups, grouped


def _summarize_counts(counts, od_groups, *, interior=False):
    rows = len(counts)
    tp, fp, fn = counts.T
    predicted, truth = tp+fp, tp+fn
    nonempty = truth > 0
    empty = ~nonempty
    _, grouped = _group_counts(counts, od_groups)
    macro = _macro(counts, nonempty if interior else None)
    group_nonempty = grouped[:, 0]+grouped[:, 2] > 0
    point = _micro(counts)
    point.update(rows=rows, point_macro=macro,
                 od_group_macro=_macro(grouped, group_nonempty if interior else None),
                 macro_population='GT-nonempty only' if interior else 'all queries / all OD groups',
                 nonempty_gt_micro=_micro(counts[nonempty]),
                 full_gt_coverage=dict(all_queries=float(np.mean(fn == 0)),
                                       nonempty_gt=float(np.mean(fn[nonempty] == 0)) if nonempty.any() else None,
                                       covered_queries=int(np.sum(fn == 0)),
                                       nonempty_gt_queries=int(nonempty.sum())),
                 region=dict(mean_predicted_cells=float(predicted.mean()),
                             mean_fp_cells=float(fp.mean()),
                             p50_predicted_cells=float(np.quantile(predicted, .5)),
                             p95_predicted_cells=float(np.quantile(predicted, .95)),
                             max_predicted_cells=int(predicted.max())),
                 empty_cases=dict(gt_empty_queries=int(empty.sum()),
                                  predicted_empty_queries=int((predicted == 0).sum()),
                                  both_empty_queries=int((empty & (predicted == 0)).sum()),
                                  gt_empty_with_false_positives=int((empty & (fp > 0)).sum()),
                                  false_positive_cells_on_empty_gt=int(fp[empty].sum(dtype=np.int64)),
                                  mean_fp_on_empty_gt=float(fp[empty].mean()) if empty.any() else None))
    return point


def metrics_at_threshold(scores, gt, endpoints, road_support, od_groups, threshold):
    scores, gt, endpoints, road_support = _validate_arrays(scores, gt, endpoints, road_support)
    if not np.isfinite(threshold) or not (0 <= threshold <= 1 or threshold == EMPTY_THRESHOLD):
        raise ValueError('threshold must be [0,1] or the frozen empty-core boundary')
    od_groups = np.asarray(od_groups, dtype=np.int64)
    if od_groups.shape != (len(scores),):
        raise ValueError('one canonical OD-group identity is required per query')
    policies = ('raw', 'handoff', 'endpoint_only', 'interior')
    counts = {policy: np.empty((len(scores), 3), dtype=np.int64) for policy in policies}
    unsupported_gt = 0
    endpoint_outside_gt = 0
    for begin in range(0, len(scores), 1024):
        end = min(begin+1024, len(scores))
        truth = gt[begin:end]
        endpoint = _endpoint_mask(endpoints[begin:end], scores.shape[1])
        # Use float64 comparison to preserve the exact recorded scalar boundary.
        raw = np.asarray(scores[begin:end], dtype=np.float64) >= float(threshold)
        handoff = (raw & road_support[None, :]) | endpoint
        counts['raw'][begin:end] = _count_rows(raw, truth)
        counts['handoff'][begin:end] = _count_rows(handoff, truth)
        counts['endpoint_only'][begin:end] = _count_rows(endpoint, truth)
        counts['interior'][begin:end] = _count_rows(handoff & ~endpoint, truth & ~endpoint)
        unsupported_gt += int(np.count_nonzero(truth & ~road_support[None, :]))
        endpoint_outside_gt += int(np.count_nonzero(endpoint & ~truth))
    diagnostics = {policy: _summarize_counts(counts[policy], od_groups, interior=policy == 'interior')
                   for policy in policies}
    diagnostics['gt_audit'] = dict(gt_cells_outside_road_support=unsupported_gt,
                                   known_endpoint_cells_outside_gt=endpoint_outside_gt,
                                   gt_modified=False)
    return diagnostics, counts


def bootstrap_by_od(counts, od_groups, *, seed=20260913, resamples=2000):
    if resamples != 2000:
        raise ValueError('final protocol requires exactly 2000 OD-group resamples')
    groups, grouped = _group_counts(np.asarray(counts, dtype=np.int64), od_groups)
    if len(groups) == 0:
        raise ValueError('bootstrap requires at least one complete OD group')
    rng = np.random.Generator(np.random.PCG64(seed))
    results = np.empty((resamples, 4), dtype=np.float64)
    for begin in range(0, resamples, 32):
        length = min(32, resamples-begin)
        sampled = rng.integers(0, len(groups), size=(length, len(groups)))
        totals = grouped[sampled].sum(axis=1, dtype=np.int64)
        tp, fp, fn = totals.T
        recall, precision, f1 = _ratios(tp, tp+fp, tp+fn)
        results[begin:begin+length] = np.column_stack((recall, precision, f1, np.minimum(recall, f1)))
    intervals = {}
    for index, metric in enumerate(('recall_micro', 'precision_micro', 'f1_micro', 'q')):
        lower, upper = np.percentile(results[:, index], [2.5, 97.5])
        intervals[metric] = [float(lower), float(upper)]
    for name in ('recall_micro', 'f1_micro', 'q'):
        intervals[name+'_minus_target'] = [value-TARGET for value in intervals[name]]
    return dict(unit='canonical OD group; all member queries retained',
                resamples=resamples, seed=int(seed), groups=len(groups),
                interval='percentile 95%', intervals=intervals,
                point_estimate=_micro(counts), target=TARGET,
                acceptance_uses_point_estimate=True,
                confidence_lower_bound_is_not_an_extra_gate=True)


def _new_audit():
    return dict(values=0, finite=0, nan=0, positive_infinity=0,
                negative_infinity=0, below_minus_one=0, above_one=0,
                unclipped_finite_min=None, unclipped_finite_max=None)


def _update_audit(audit, values):
    values = np.asarray(values)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    audit['values'] += values.size
    audit['finite'] += int(finite_mask.sum())
    audit['nan'] += int(np.isnan(values).sum())
    audit['positive_infinity'] += int(np.isposinf(values).sum())
    audit['negative_infinity'] += int(np.isneginf(values).sum())
    audit['below_minus_one'] += int(np.count_nonzero(finite < -1))
    audit['above_one'] += int(np.count_nonzero(finite > 1))
    if finite.size:
        low, high = float(finite.min()), float(finite.max())
        audit['unclipped_finite_min'] = low if audit['unclipped_finite_min'] is None else min(low, audit['unclipped_finite_min'])
        audit['unclipped_finite_max'] = high if audit['unclipped_finite_max'] is None else max(high, audit['unclipped_finite_max'])
    return bool(finite_mask.all())


