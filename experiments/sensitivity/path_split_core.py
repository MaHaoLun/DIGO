"""Outcome-blind path-derived identity construction; definitions only.

Importing this module does not read a graph, enumerate real parents, create
cases, draw random numbers, or call a predictor/solver.  An authorized metadata
entry must load the separately bound training roles and parent witnesses, then
call these functions after the new study's analysis gate has passed.

Only route node identities enter this interface.  Parent cost, departure,
arrivals, predictions, search timing, and regional answers are not inputs.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Sequence, Tuple


SETTINGS = (("BJ", "G0"), ("XA", "G0"), ("BJ", "G1"), ("XA", "G1"))
EXPECTED_PARENT_COUNTS = {("BJ", "G0"): 800, ("XA", "G0"): 344,
                          ("BJ", "G1"): 2971, ("XA", "G1"): 1160}
WINDOWS = {"G0": (28800, 43200), "G1": (43200, 57600)}
ARMS = ("full_graph", "union_ges", "single_expansion", "union_expansion")
TARGETS = (100, 104, 108)
SEED_PREFIX = (20261102, 12)


class SourceContractError(RuntimeError):
    """A missing/mismatched source blocks construction; it is not no-child."""


@dataclass(frozen=True)
class Node:
    index: int
    cell64: int


@dataclass(frozen=True)
class TrainingSupport:
    city: str
    snapshot: str
    node_count: int
    distances64: FrozenSet[int]
    source_node_ids: FrozenSet[int]
    destination_node_ids: FrozenSet[int]
    # Actual fit/calibration/selection/development identities, decoded before
    # this interface. E1 groups[row_index] must never be decoded by divmod(N).
    non_test_pairs: Mapping[Tuple[int, int], Tuple[str, ...]]
    historical_test_pairs: FrozenSet[Tuple[int, int]]
    model: Mapping
    graph: Mapping


@dataclass(frozen=True)
class Parent:
    city: str
    snapshot: str
    serial: int
    source: int
    destination: int
    route: Tuple[int, ...]  # Original node IDs, in saved witness order.
    interval: Tuple[int, int]
    model: Mapping
    graph: Mapping
    route_source: Mapping
    audit_source: Mapping
    accepted_full_witness: bool


@dataclass(frozen=True)
class SplitConfig:
    targets64: Tuple[int, int, int]
    seed_prefix: Tuple[int, ...]
    per_stratum_cap: int = 128


def distance64(a: Node, b: Node) -> int:
    for node in (a, b):
        if not (0 <= node.cell64 < 4096 and node.index >= 0):
            raise SourceContractError("Unknown or invalid actual 64-grid node")
    return abs(a.cell64 % 64 - b.cell64 % 64) + abs(a.cell64 // 64 - b.cell64 // 64)


def _key(row):
    return (row["city"], row["snapshot"], row["interval"][0], row["interval"][1],
            row["source"], row["destination"])


def _provenance_key(row):
    return row["parent_serial"], row["start_position"], row["end_position"]


def _config_record(config):
    return dict(targets64=list(config.targets64), seed_prefix=list(config.seed_prefix),
        per_stratum_cap=config.per_stratum_cap,
        unique_directed_od_before_uniform_sampling=True,
        stable_registration_order_after_uniform_subset=True,
        reverse_is_a_distinct_input=True)


def _validate(parent, support, nodes, config):
    if (parent.city, parent.snapshot) != (support.city, support.snapshot):
        raise SourceContractError("Parent and fixed-model support differ")
    if parent.interval != WINDOWS[parent.snapshot]:
        raise SourceContractError("Parent snapshot/window cannot be relabelled")
    if parent.model != support.model or parent.graph != support.graph:
        raise SourceContractError("Parent graph/model is not the bound original setting")
    if parent.accepted_full_witness is not True:
        raise SourceContractError("An accepted original Full witness is required")
    if not parent.route_source or not parent.audit_source:
        raise SourceContractError("Missing saved witness/audit provenance")
    if len(parent.route) < 2 or (parent.route[0], parent.route[-1]) != (parent.source, parent.destination):
        raise SourceContractError("Saved witness endpoint mismatch")
    if any(node_id not in nodes for node_id in parent.route):
        raise SourceContractError("Unknown saved witness node ID")
    if config.targets64 != TARGETS:
        raise SourceContractError("The frozen three in-support points are 100, 104, 108")
    if tuple(sorted(config.targets64)) != config.targets64 or config.per_stratum_cap != 128:
        raise SourceContractError("Changed ordered points or fixed maximum")
    if any(d not in support.distances64 for d in config.targets64):
        raise SourceContractError("A child target is outside this model's actual training distance support")
    if config.seed_prefix != SEED_PREFIX:
        raise SourceContractError("The published plan freezes seed prefix [20261102,12]")


def enumerate_parent(parent: Parent, nodes: Mapping[int, Node], support: TrainingSupport,
                     config: SplitConfig, all_parent_pairs: FrozenSet[Tuple[int, int]]):
    """Enumerate proper directed subpaths using identities and roles only.

    Qualification is applied to each child. A lawful parent does not grant its
    internal nodes/ODs training support or an evaluation role. The parent path
    proves only an ordered identity source, never child full-window optimality.
    """
    _validate(parent, support, nodes, config)
    route = parent.route
    parent_distance = distance64(nodes[parent.source], nodes[parent.destination])
    receipts = {d: dict(city=parent.city, snapshot=parent.snapshot,
        parent_serial=parent.serial, target_distance64=d,
        parent_actual_distance64=parent_distance,
        route_source=dict(parent.route_source), audit_source=dict(parent.audit_source),
        route_node_count=len(route), proper_position_pairs=len(route) * (len(route) - 1) // 2 - 1,
        supported_endpoint_position_pairs=0, matching_distance_position_pairs=0,
        excluded_matching_position_pairs=0,
        locally_eligible_position_pairs=0, locally_eligible_unique_directed_od=0,
        rejection_counts={}, rejection_reasons_may_overlap=True,
        no_child_reason=None) for d in config.targets64}
    rejects = {d: Counter() for d in config.targets64}
    eligible = {d: set() for d in config.targets64}
    candidates = []
    source_positions = [i for i, node_id in enumerate(route[:-1]) if node_id in support.source_node_ids]
    destination_positions = [j for j, node_id in enumerate(route) if j > 0 and node_id in support.destination_node_ids]
    for i in source_positions:
        for j in destination_positions:
            if j <= i or (i == 0 and j == len(route) - 1):
                continue
            for receipt in receipts.values():
                receipt["supported_endpoint_position_pairs"] += 1
            source, destination = route[i], route[j]
            d = distance64(nodes[source], nodes[destination])
            if d not in receipts:
                continue
            receipts[d]["matching_distance_position_pairs"] += 1
            pair, reverse = (source, destination), (destination, source)
            reasons = []
            if source == destination:
                reasons.append("DEGENERATE_SAME_NODE")
            if pair == (parent.source, parent.destination):
                reasons.append("SAME_PHYSICAL_OD_AS_OWN_PARENT")
            blocked = tuple(support.non_test_pairs.get(pair, ())) + tuple(support.non_test_pairs.get(reverse, ()))
            if blocked:
                reasons.extend("NON_TEST_OR_REVERSE:" + reason for reason in sorted(set(blocked)))
            if reasons:
                receipts[d]["excluded_matching_position_pairs"] += 1
                rejects[d].update(reasons)
                continue
            ns, nt = nodes[source], nodes[destination]
            if max(ns.index, nt.index) >= support.node_count:
                raise SourceContractError("Node index outside the original graph")
            also_parent = pair in all_parent_pairs or reverse in all_parent_pairs
            historical = also_parent or pair in support.historical_test_pairs or reverse in support.historical_test_pairs
            candidates.append(dict(city=parent.city, snapshot=parent.snapshot,
                exact_distance64=d, interval=list(parent.interval),
                source=source, destination=destination,
                source_index=ns.index, destination_index=nt.index,
                endpoint_cells64=[ns.cell64, nt.cell64],
                physical_pair_code=ns.index * support.node_count + nt.index,
                # Keep G0's legacy groups-row identifier separate; it is never
                # inherited from the parent or inferred from this pair code.
                legacy_model_group_id=(ns.index * support.node_count + nt.index
                                       if parent.snapshot == "G1" else None),
                legacy_encoding_kind=("E1_groups_row_requires_explicit_reverse_lookup"
                                      if parent.snapshot == "G0" else "source_index_times_n_plus_destination_index"),
                virtual_child_identity="selected_cohort_distance_v12:{}:{}:{}:{}".format(
                    parent.city, parent.snapshot, source, destination),
                neural_noise_recipe_not_frozen_by_identity_construction=True,
                exposure_class="historical_test_only_not_fresh" if historical else "not_in_named_non_test_or_test_sources",
                also_in_original_parent_cohort_or_reverse=also_parent,
                model=dict(support.model), graph=dict(support.graph),
                provenance=dict(parent_serial=parent.serial, start_position=i, end_position=j,
                    parent_actual_distance64=parent_distance,
                    parent_witness_arc_count=len(route) - 1, child_witness_arc_count=j - i,
                    child_geometric_distance_relative_to_parent=("smaller" if d < parent_distance
                        else "equal" if d == parent_distance else "greater"),
                    parent_route_source=dict(parent.route_source), parent_audit_source=dict(parent.audit_source))))
            receipts[d]["locally_eligible_position_pairs"] += 1
            eligible[d].add(pair)
    for d, receipt in receipts.items():
        receipt["locally_eligible_unique_directed_od"] = len(eligible[d])
        receipt["rejection_counts"] = dict(sorted(rejects[d].items()))
        if receipt["excluded_matching_position_pairs"] + receipt["locally_eligible_position_pairs"] != receipt["matching_distance_position_pairs"]:
            raise SourceContractError("Every matching child position pair must be retained or explicitly excluded")
        if receipt["supported_endpoint_position_pairs"] == 0:
            receipt["no_child_reason"] = "NO_SUPPORTED_ENDPOINT_PROPER_SUBPATH"
        elif receipt["matching_distance_position_pairs"] == 0:
            receipt["no_child_reason"] = "NO_TARGET_DISTANCE_WITH_SUPPORTED_ENDPOINTS"
        elif not eligible[d]:
            receipt["no_child_reason"] = "ALL_TARGET_PAIRS_EXCLUDED_SEE_COUNTS"
        else:
            receipt["no_child_reason"] = "LOCALLY_ELIGIBLE_BEFORE_GLOBAL_DEDUP_AND_FIXED_CAP"
    return candidates, [receipts[d] for d in config.targets64]


def collect_unique_pool(parents: Sequence[Parent], nodes_by_city, supports, config: SplitConfig):
    """Cover all 5275 parents, preserve all same-OD witness sources, no draws."""
    if Counter((p.city, p.snapshot) for p in parents) != Counter(EXPECTED_PARENT_COUNTS):
        raise SourceContractError("All four original parent denominators are required")
    if len({(p.city, p.snapshot, p.serial) for p in parents}) != 5275:
        raise SourceContractError("Parent serial identity duplicate or missing")
    parent_pairs = {city: frozenset((p.source, p.destination) for p in parents if p.city == city)
                    for city in ("BJ", "XA")}
    pool, parent_receipts, source_errors = {}, [], []
    for parent in sorted(parents, key=lambda p: (p.serial, p.city, p.snapshot)):
        try:
            candidates, receipts = enumerate_parent(parent, nodes_by_city[parent.city],
                supports[parent.city, parent.snapshot], config, parent_pairs[parent.city])
        except SourceContractError as error:
            # Complete the parent denominator even when one source is bad.
            # The entire pool remains blocked; this never becomes a normal
            # no-child exclusion or a smaller successful cohort.
            source_errors.append(dict(city=parent.city, snapshot=parent.snapshot,
                parent_serial=parent.serial, error=str(error)))
            parent_receipts.extend(dict(city=parent.city, snapshot=parent.snapshot,
                parent_serial=parent.serial, target_distance64=d,
                no_child_reason="SOURCE_CONTRACT_ERROR_BLOCKS_ALL_CONSTRUCTION",
                error=str(error)) for d in config.targets64)
            continue
        parent_receipts.extend(receipts)
        for candidate in candidates:
            key = _key(candidate)
            provenance = candidate.pop("provenance")
            if key not in pool:
                pool[key] = dict(candidate, parent_provenance=[])
            else:
                for field in ("exact_distance64", "source_index", "destination_index", "model", "graph"):
                    if pool[key][field] != candidate[field]:
                        raise SourceContractError("Same child identity has conflicting metadata")
            pool[key]["parent_provenance"].append(provenance)
    for row in pool.values():
        row["parent_provenance"].sort(key=_provenance_key)
    reverse_links = []
    for key in sorted(pool):
        reverse_key = key[:4] + (key[5], key[4])
        if reverse_key in pool and key < reverse_key:
            reverse_links.append(dict(identity=list(key), reverse_identity=list(reverse_key),
                reason="DISTINCT_DIRECTED_QUERY_INPUTS_NEVER_A_FREE_ALIAS"))
    if len(parent_receipts) != 15825:
        raise SourceContractError("Every parent-target receipt is required")
    return dict(unique_pool=[pool[k] for k in sorted(pool)], parent_target_ledger=parent_receipts,
        reverse_query_linkage=reverse_links, source_contract_pass=not source_errors,
        source_errors=source_errors, expected_parent_queries=5275,
        construction_config=_config_record(config),
        parent_target_ledger_rows=15825, new_model_calls=0, new_search_calls=0,
        parent_cost_departure_score_or_runtime_used=False)


def select_fixed_prefix(pool_result, config: SplitConfig):
    """One fixed PCG64 draw from each complete unique pool, never outcomes.

    The authorized caller must bind NumPy's actual version and this function's
    source before calling it. There is no early stopping, replacement after an
    answer, or filling a sparse target using other parents/populations.
    """
    if pool_result["source_contract_pass"] is not True:
        raise SourceContractError("Source errors cannot be downgraded to a successful smaller pool")
    if pool_result["construction_config"] != _config_record(config):
        raise SourceContractError("Collection and selection must use the same bound plan configuration")
    if config.seed_prefix != SEED_PREFIX or config.targets64 != TARGETS or config.per_stratum_cap != 128:
        raise SourceContractError("Changed published plan configuration")
    if len(pool_result["parent_target_ledger"]) != 15825:
        raise SourceContractError("Incomplete parent-target coverage ledger")
    import numpy as np
    by_stratum = defaultdict(list)
    for row in pool_result["unique_pool"]:
        by_stratum[row["city"], row["snapshot"], row["exact_distance64"]].append(row)
    selected, selection_receipts = [], []
    for city, snapshot in SETTINGS:
        for target in config.targets64:
            rows = sorted(by_stratum[city, snapshot, target], key=_key)
            entropy = list(config.seed_prefix) + [1 if city == "BJ" else 0, 0 if snapshot == "G0" else 1, target]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
            order = [int(i) for i in rng.permutation(len(rows))]
            chosen = order[:min(config.per_stratum_cap, len(rows))]
            draw_rank = {index: rank for rank, index in enumerate(chosen, 1)}
            registration_order = sorted(chosen)
            for rank, index in enumerate(registration_order, 1):
                row = deepcopy(rows[index])
                row.update(case_id="{}_{}_d64_{}_ps{:03d}".format(city, snapshot, target, rank),
                    selection_rank=rank, draw_rank=draw_rank[index], pool_index=index, sampling_seed_sequence=entropy,
                    query_arms=list(ARMS), parent_witness_used_for_identity_only=True,
                    requires_own_independently_computed_4h_reference=True,
                    parent_answer_seed_prediction_and_cost_not_reused_as_child=True)
                selected.append(row)
            selection_receipts.append(dict(city=city, snapshot=snapshot, exact_distance64=target,
                qualified_unique_pool=len(rows), selected_N=len(chosen), maximum_N=128,
                seed_sequence=entropy, full_permutation=order, all_if_below_maximum=True,
                selected_pool_indices_in_draw_order=chosen,
                selected_pool_indices_in_registration_order=registration_order,
                empty_target_retained=True, no_outcome_filter_or_replacement=True))
    pool_by_parent, selected_by_parent = defaultdict(set), defaultdict(set)
    for rows, mapping in ((pool_result["unique_pool"], pool_by_parent), (selected, selected_by_parent)):
        for row in rows:
            for origin in row["parent_provenance"]:
                pkey = row["city"], row["snapshot"], origin["parent_serial"], row["exact_distance64"]
                mapping[pkey].add(_key(row))
    receipts = deepcopy(pool_result["parent_target_ledger"])
    for receipt in receipts:
        pkey = receipt["city"], receipt["snapshot"], receipt["parent_serial"], receipt["target_distance64"]
        receipt["globally_unique_child_candidates"] = len(pool_by_parent[pkey])
        receipt["selected_unique_children"] = len(selected_by_parent[pkey])
        if receipt["globally_unique_child_candidates"]:
            receipt["no_child_reason"] = ("HAS_FIXED_SELECTED_CHILD" if receipt["selected_unique_children"]
                                           else "NO_CHILD_SELECTED_BY_FIXED_UNIFORM_CAP")
    return dict(selected_children=selected, selection_receipts=selection_receipts,
        parent_target_ledger=receipts, reverse_query_linkage=pool_result["reverse_query_linkage"],
        fixed_maximum_children=1536, fixed_maximum_logical_rows=6144,
        numpy_version=np.__version__, bit_generator="PCG64",
        new_model_calls=0, new_search_calls=0, new_masks=0,
        not_permission_for_neural_or_search_execution=True)


if __name__ == "__main__":
    raise SystemExit("Definitions only: real child construction requires the new study analysis gate and bound metadata entry")
