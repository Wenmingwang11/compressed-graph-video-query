from __future__ import annotations

from collections import Counter
from itertools import product
import time
from typing import Mapping, MutableMapping, Sequence, Set

import igraph as ig
import pandas as pd

from supplement.corrected_query_model import QueryResult, QuerySpec, exact_query
from vsimsearch.paper_querying import (
    _filter_window_ids_by_types,
    _invert_window_type_index,
    _vertex_covers_query_types,
    iter_bindings_in_vertex,
)


def _collect_directed_pair_index(
    graph: ig.Graph,
    vertex_idx: int,
    pair: tuple[int, int],
    cache: MutableMapping[tuple[int, tuple[int, int]], dict[tuple[int, int], set[int]]],
) -> dict[tuple[int, int], set[int]]:
    cache_key = (vertex_idx, pair)
    if cache_key in cache:
        return cache[cache_key]

    merged: dict[tuple[int, int], set[int]] = {}
    visited: set[int] = set()

    def visit(current_idx: int) -> None:
        if current_idx in visited:
            return
        visited.add(current_idx)
        combinations = graph.vs[current_idx]["combinations"] or {}
        for (theta_bin, distance_bin), frames in (combinations.get(pair) or {}).items():
            merged.setdefault((int(theta_bin), int(distance_bin)), set()).update(
                int(frame) for frame in frames
            )
        for predecessor in graph.predecessors(current_idx):
            visit(predecessor)

    visit(vertex_idx)
    cache[cache_key] = merged
    return merged


def _window_candidates(
    graph: ig.Graph,
    window_data: Mapping[int, Set[int]],
    query: QuerySpec,
    actual_scale: int,
    indexed_scale: int,
    legacy_single_direction: bool,
) -> dict[tuple[int, ...], set[int]]:
    id_to_type = _invert_window_type_index(window_data)
    required_counts = Counter(query.role_types)
    candidate_vertices: list[tuple[int, int, tuple[int, ...]]] = []
    for vertex_idx, vertex in enumerate(graph.vs):
        node_ids = tuple(int(node_id) for node_id in vertex["name"])
        if len(node_ids) >= len(query.role_types) and _vertex_covers_query_types(
            node_ids, id_to_type, required_counts
        ):
            candidate_vertices.append((len(node_ids), vertex_idx, node_ids))
    candidate_vertices.sort(key=lambda item: (item[0], item[1]))

    cache: dict[tuple[int, tuple[int, int]], dict[tuple[int, int], set[int]]] = {}
    results: dict[tuple[int, ...], set[int]] = {}
    seen_bindings: set[tuple[int, ...]] = set()
    for _, vertex_idx, node_ids in candidate_vertices:
        for binding in iter_bindings_in_vertex(node_ids, id_to_type, query.role_types):
            normalized = tuple(int(track_id) for track_id in binding)
            if normalized in seen_bindings:
                continue
            seen_bindings.add(normalized)

            relation_frames: list[set[int]] = []
            for (source_role, target_role), constraint in query.relations.items():
                source_id = normalized[source_role]
                target_id = normalized[target_role]
                direct_index = _collect_directed_pair_index(
                    graph,
                    vertex_idx,
                    (source_id, target_id),
                    cache,
                )
                matched: set[int] = set()
                for (theta_bin, distance_bin), frames in direct_index.items():
                    if theta_bin in constraint.theta_bins and _distance_intervals_overlap(
                        distance_bin,
                        constraint.d_min,
                        constraint.d_max,
                        indexed_scale,
                        actual_scale,
                        query.distance_parts,
                    ):
                        matched.update(frames)
                if legacy_single_direction:
                    reverse_index = _collect_directed_pair_index(
                        graph,
                        vertex_idx,
                        (target_id, source_id),
                        cache,
                    )
                    for (theta_bin, distance_bin), frames in reverse_index.items():
                        directed_theta = theta_bin - query.theta_parts if theta_bin >= 0 else theta_bin + query.theta_parts
                        if directed_theta in constraint.theta_bins and _distance_intervals_overlap(
                            distance_bin,
                            constraint.d_min,
                            constraint.d_max,
                            indexed_scale,
                            actual_scale,
                            query.distance_parts,
                        ):
                            matched.update(frames)
                if not matched:
                    relation_frames = []
                    break
                relation_frames.append(matched)
            if not relation_frames:
                continue
            common = set.intersection(*relation_frames)
            if common:
                results[normalized] = common
    return results


def _distance_intervals_overlap(
    indexed_bin: int,
    query_d_min: int,
    query_d_max: int,
    indexed_scale: int,
    actual_scale: int,
    distance_parts: int,
) -> bool:
    indexed_low = indexed_bin * indexed_scale / distance_parts
    indexed_high = (indexed_bin + 1) * indexed_scale / distance_parts
    query_low = query_d_min * actual_scale / distance_parts
    query_high = (query_d_max + 1) * actual_scale / distance_parts
    return indexed_low < query_high and query_low < indexed_high


def _flatten_explicit_pair_indexes(
    graph: ig.Graph,
) -> dict[tuple[int, int], dict[tuple[int, int], set[int]]]:
    flattened: dict[tuple[int, int], dict[tuple[int, int], set[int]]] = {}
    for vertex in graph.vs:
        for pair, theta_distance_map in (vertex["combinations"] or {}).items():
            pair_key = (int(pair[0]), int(pair[1]))
            target = flattened.setdefault(pair_key, {})
            for (theta_bin, distance_bin), frames in theta_distance_map.items():
                target.setdefault((int(theta_bin), int(distance_bin)), set()).update(
                    int(frame) for frame in frames
                )
    return flattened


def _relation_assignments(
    pair_indexes: Mapping[tuple[int, int], Mapping[tuple[int, int], Set[int]]],
    id_to_type: Mapping[int, Set[int]],
    source_role: int,
    target_role: int,
    source_type: int,
    target_type: int,
    constraint,
    actual_scale: int,
    indexed_scale: int,
    distance_parts: int,
    theta_parts: int,
    legacy_single_direction: bool,
) -> list[tuple[dict[int, int], set[int]]]:
    assignments: list[tuple[dict[int, int], set[int]]] = []
    for (stored_source, stored_target), theta_distance_map in pair_indexes.items():
        stored_source_types = id_to_type.get(stored_source, set())
        stored_target_types = id_to_type.get(stored_target, set())
        orientations: list[tuple[int, int, bool]] = []
        if source_type in stored_source_types and target_type in stored_target_types:
            orientations.append((stored_source, stored_target, False))
        if legacy_single_direction and target_type in stored_source_types and source_type in stored_target_types:
            orientations.append((stored_target, stored_source, True))
        for source_id, target_id, reverse in orientations:
            matched_frames: set[int] = set()
            for (theta_bin, distance_bin), frames in theta_distance_map.items():
                directed_theta = (
                    theta_bin - theta_parts if reverse and theta_bin >= 0 else
                    theta_bin + theta_parts if reverse else
                    theta_bin
                )
                angle_matches = directed_theta in constraint.theta_bins
                # Bin zero includes an exactly horizontal vector. Its reverse
                # can be +pi as well as -pi; retain both wrap endpoints here.
                if reverse and theta_bin == 0 and theta_parts in constraint.theta_bins:
                    angle_matches = True
                if angle_matches and _distance_intervals_overlap(
                    distance_bin,
                    constraint.d_min,
                    constraint.d_max,
                    indexed_scale,
                    actual_scale,
                    distance_parts,
                ):
                    matched_frames.update(frames)
            if matched_frames:
                assignments.append(
                    ({source_role: source_id, target_role: target_id}, matched_frames)
                )
    return assignments


def _all_observed_types(window_data: Mapping[int, Set[int]]) -> dict[int, set[int]]:
    """An ID can have several observed classes; the verifier checks each frame."""
    result: dict[int, set[int]] = {}
    for class_id, object_ids in window_data.items():
        for object_id in object_ids:
            result.setdefault(int(object_id), set()).add(int(class_id))
    return result


def _window_candidates_flattened(
    graph: ig.Graph,
    window_data: Mapping[int, Set[int]],
    query: QuerySpec,
    actual_scale: int,
    indexed_scale: int,
    legacy_single_direction: bool,
) -> dict[tuple[int, ...], set[int]]:
    id_to_type = _all_observed_types(window_data)
    pair_indexes = _flatten_explicit_pair_indexes(graph)
    return _window_candidates_from_pair_indexes(
        pair_indexes,
        id_to_type,
        query,
        actual_scale,
        indexed_scale,
        legacy_single_direction,
    )


def _window_candidates_from_pair_indexes(
    pair_indexes: Mapping[tuple[int, int], Mapping[tuple[int, int], Set[int]]],
    id_to_type: Mapping[int, Set[int]],
    query: QuerySpec,
    actual_scale: int,
    indexed_scale: int,
    legacy_single_direction: bool,
) -> dict[tuple[int, ...], set[int]]:
    joined: list[tuple[dict[int, int], set[int]]] | None = None
    for (source_role, target_role), constraint in sorted(query.relations.items()):
        candidates = _relation_assignments(
            pair_indexes,
            id_to_type,
            source_role,
            target_role,
            query.role_types[source_role],
            query.role_types[target_role],
            constraint,
            actual_scale,
            indexed_scale,
            query.distance_parts,
            query.theta_parts,
            legacy_single_direction,
        )
        if not candidates:
            return {}
        if joined is None:
            joined = candidates
            continue
        next_joined: list[tuple[dict[int, int], set[int]]] = []
        for assignment, frames in joined:
            for candidate_assignment, candidate_frames in candidates:
                shared_roles = set(assignment).intersection(candidate_assignment)
                if any(assignment[role] != candidate_assignment[role] for role in shared_roles):
                    continue
                merged = {**assignment, **candidate_assignment}
                if len(set(merged.values())) != len(merged):
                    continue
                common_frames = frames.intersection(candidate_frames)
                if common_frames:
                    next_joined.append((merged, common_frames))
        joined = next_joined
        if not joined:
            return {}

    results: dict[tuple[int, ...], set[int]] = {}
    for assignment, frames in joined or []:
        if len(assignment) != len(query.role_types):
            continue
        binding = tuple(assignment[role] for role in range(len(query.role_types)))
        results.setdefault(binding, set()).update(frames)
    return results


def query_corrected_mli(
    index: Mapping[int, Mapping[int, Set[int]]],
    cp_graphs: Mapping[int, ig.Graph],
    data: pd.DataFrame,
    query: QuerySpec,
    frame_width: int,
    frame_height: int,
    legacy_single_direction: bool = False,
    indexed_frame_width: int | None = None,
    indexed_frame_height: int | None = None,
    window_edge_indexes: Mapping[
        int, Mapping[tuple[int, int], Mapping[tuple[int, int], Set[int]]]
    ]
    | None = None,
    *,
    stats: dict | None = None,
    return_all: bool = False,
) -> list[QueryResult]:
    started = time.perf_counter() if stats is not None else 0.0
    candidate_bindings: set[tuple[int, ...]] = set()
    candidate_frames: set[int] = set()
    window_ids = _filter_window_ids_by_types(index, query.role_types)
    if stats is not None:
        stats['windows_total'] = len(index)
        stats['windows_after_type_filter'] = len(window_ids)
    for window_id in window_ids:
        graph = cp_graphs.get(window_id)
        if graph is None:
            continue
        indexed_width = frame_width if indexed_frame_width is None else int(indexed_frame_width)
        indexed_height = frame_height if indexed_frame_height is None else int(indexed_frame_height)
        window_results = _window_candidates_flattened(
            graph,
            index[window_id],
            query,
            actual_scale=int(frame_width) + int(frame_height),
            indexed_scale=indexed_width + indexed_height,
            legacy_single_direction=legacy_single_direction,
        )
        if window_edge_indexes is not None and window_id in window_edge_indexes:
            id_to_type = _all_observed_types(index[window_id])
            lossless_results = _window_candidates_from_pair_indexes(
                window_edge_indexes[window_id],
                id_to_type,
                query,
                actual_scale=int(frame_width) + int(frame_height),
                indexed_scale=indexed_width + indexed_height,
                legacy_single_direction=legacy_single_direction,
            )
            for binding, frames in lossless_results.items():
                window_results.setdefault(binding, set()).update(frames)
            if len(query.role_types) >= 3:
                role_candidates = [
                    sorted(index[window_id].get(int(role_type), set()))
                    for role_type in query.role_types
                ]
                if all(role_candidates):
                    for binding in product(*role_candidates):
                        normalized = tuple(int(track_id) for track_id in binding)
                        if len(set(normalized)) == len(normalized):
                            candidate_bindings.add(normalized)
                for pair_map in window_edge_indexes[window_id].values():
                    for frames in pair_map.values():
                        candidate_frames.update(int(frame) for frame in frames)
        for binding, frames in window_results.items():
            candidate_bindings.add(binding)
            candidate_frames.update(frames)
    if stats is not None:
        stats['candidate_s'] = time.perf_counter() - started
        stats['candidate_distinct_bindings'] = len(candidate_bindings)
    if (not candidate_bindings or not candidate_frames) and stats is None:
        return []
    return exact_query(
        data,
        query,
        frame_width,
        frame_height,
        candidate_bindings=candidate_bindings,
        candidate_frames=candidate_frames,
        stats=stats,
        return_all=return_all,
    )


__all__ = ["query_corrected_mli"]
