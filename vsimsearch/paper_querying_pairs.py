from __future__ import annotations

from itertools import combinations
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

import igraph as ig

from vsimsearch.paper_querying import (
    D,
    FrameId,
    NodeId,
    NodeType,
    PaperQueryStats,
    PredecessorRelationResolver,
    Theta,
    WindowId,
    WindowTypeIndex,
    _filter_window_ids_by_types,
    _invert_window_type_index,
    _theta_d_satisfy,
    _vertex_covers_query_types,
    iter_bindings_in_vertex,
)


RolePair = Tuple[int, int]
PairConstraintMap = Mapping[RolePair, Tuple[Theta, D]]


def parse_pair_constraints(text: str, role_count: int) -> Dict[RolePair, Tuple[Theta, D]]:
    if role_count < 2:
        return {}
    if not text.strip():
        raise ValueError("pair_constraints must not be empty")

    constraints: Dict[RolePair, Tuple[Theta, D]] = {}
    parts = [part.strip() for part in text.split(";") if part.strip()]
    for part in parts:
        try:
            role_part, condition_part = part.split(":", 1)
            role_a_text, role_b_text = role_part.split("-", 1)
            theta_text, d_text = condition_part.split(",", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid pair constraint '{part}', expected format like 0-1:1.0,5"
            ) from exc

        role_a = int(role_a_text.strip())
        role_b = int(role_b_text.strip())
        if role_a == role_b:
            raise ValueError(f"invalid role pair '{role_part}': roles must be different")
        if role_a < 0 or role_b < 0 or role_a >= role_count or role_b >= role_count:
            raise ValueError(f"invalid role pair '{role_part}': role index out of range")

        role_pair = tuple(sorted((role_a, role_b)))
        if role_pair in constraints:
            raise ValueError(f"duplicate pair constraint for role pair {role_pair}")
        constraints[role_pair] = (float(theta_text.strip()), float(d_text.strip()))

    expected_pairs = {tuple(sorted(pair)) for pair in combinations(range(role_count), 2)}
    missing_pairs = sorted(expected_pairs - set(constraints.keys()))
    extra_pairs = sorted(set(constraints.keys()) - expected_pairs)
    if missing_pairs or extra_pairs:
        problems = []
        if missing_pairs:
            problems.append(f"missing pairs: {missing_pairs}")
        if extra_pairs:
            problems.append(f"unexpected pairs: {extra_pairs}")
        raise ValueError("; ".join(problems))

    return constraints


def match_bindings_in_window_with_pair_constraints(
    graph: ig.Graph,
    window_data: Mapping[NodeType, Set[NodeId]],
    query_object_types: Sequence[NodeType],
    pair_constraints: PairConstraintMap,
    max_bindings_per_window: int = 0,
    stats: Optional[PaperQueryStats] = None,
) -> Dict[Tuple[NodeId, ...], Set[FrameId]]:
    id_to_type = _invert_window_type_index(window_data)

    candidate_vertices: list[Tuple[int, int, Tuple[NodeId, ...]]] = []
    required_counts = {}
    for node_type in query_object_types:
        required_counts[node_type] = required_counts.get(node_type, 0) + 1
    for vertex_idx, vertex in enumerate(graph.vs):
        node_ids = tuple(vertex["name"])
        if len(node_ids) < len(query_object_types):
            continue
        if _vertex_covers_query_types(node_ids, id_to_type, required_counts):
            candidate_vertices.append((len(node_ids), vertex_idx, node_ids))

    candidate_vertices.sort(key=lambda item: (item[0], item[1]))
    if stats is not None:
        stats.candidate_vertices_total += len(candidate_vertices)

    resolver = PredecessorRelationResolver(graph)
    results: Dict[Tuple[NodeId, ...], Set[FrameId]] = {}
    seen_bindings: Set[Tuple[NodeId, ...]] = set()

    for _, vertex_idx, node_ids in candidate_vertices:
        for binding in iter_bindings_in_vertex(node_ids, id_to_type, query_object_types):
            if binding in seen_bindings:
                continue
            seen_bindings.add(binding)

            if max_bindings_per_window and len(seen_bindings) > max_bindings_per_window:
                return results

            if stats is not None:
                stats.bindings_total += 1

            per_pair_frames: list[Set[FrameId]] = []
            for role_i, role_j in combinations(range(len(binding)), 2):
                if stats is not None:
                    stats.pair_lookups += 1

                pair_index = resolver.collect_pair_index_from_predecessors(
                    vertex_idx, (binding[role_i], binding[role_j])
                )
                if not pair_index:
                    per_pair_frames = []
                    break

                theta_d_condition = pair_constraints[(role_i, role_j)]
                matched_frames: Set[FrameId] = set()
                for (theta, d), frame_set in pair_index.items():
                    if stats is not None:
                        stats.theta_d_bins_scanned += 1
                    if _theta_d_satisfy(theta, d, theta_d_condition):
                        matched_frames.update(frame_set)

                if not matched_frames:
                    per_pair_frames = []
                    break
                per_pair_frames.append(matched_frames)

            if not per_pair_frames:
                continue

            intersection = set.intersection(*per_pair_frames)
            if intersection:
                results[binding] = intersection
                if stats is not None:
                    stats.bindings_matched += 1

    return results


def run_paper_query_with_pair_constraints(
    index: WindowTypeIndex,
    cp_graphs: Mapping[WindowId, ig.Graph],
    query_object_types: Sequence[NodeType],
    pair_constraints: PairConstraintMap,
    topk: int,
    frame_threshold: int,
    max_bindings_per_window: int = 0,
):
    from prefix_tree import build_tree_from_results, get_top_five_paths_with_threshold

    stats = PaperQueryStats(windows_total=len(index))
    candidate_window_ids = _filter_window_ids_by_types(index, query_object_types)
    stats.windows_after_type_filter = len(candidate_window_ids)

    results: Dict[int, Dict[Tuple[NodeId, ...], Set[FrameId]]] = {}
    for window_id in candidate_window_ids:
        graph = cp_graphs.get(window_id)
        if graph is None:
            continue

        window_result = match_bindings_in_window_with_pair_constraints(
            graph=graph,
            window_data=index[window_id],
            query_object_types=query_object_types,
            pair_constraints=pair_constraints,
            max_bindings_per_window=max_bindings_per_window,
            stats=stats,
        )
        if window_result:
            results[window_id] = window_result

    tree_root = build_tree_from_results(results)
    top_paths = get_top_five_paths_with_threshold(tree_root, frame_threshold, topk)
    return stats, top_paths
