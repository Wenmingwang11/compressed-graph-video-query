from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import igraph as ig


WindowId = int
NodeType = int
NodeId = int
Theta = float
D = float
FrameId = int

# window_id -> node_type -> set(node_id)
WindowTypeIndex = Mapping[WindowId, Mapping[NodeType, Set[NodeId]]]


@dataclass
class PaperQueryStats:
    windows_total: int = 0
    windows_after_type_filter: int = 0
    candidate_vertices_total: int = 0
    bindings_total: int = 0
    bindings_matched: int = 0
    pair_lookups: int = 0
    theta_d_bins_scanned: int = 0


def _theta_d_satisfy(theta: Theta, d: D, condition: Optional[Tuple[Theta, D]]) -> bool:
    if condition is None:
        return True
    theta_lt, d_lt = condition
    return theta < theta_lt and d < d_lt


def _filter_window_ids_by_types(
    index: WindowTypeIndex,
    query_object_types: Sequence[NodeType],
) -> list[WindowId]:
    required = Counter(query_object_types)
    return [
        window_id
        for window_id, window_data in index.items()
        if all(len(window_data.get(node_type, set())) >= count for node_type, count in required.items())
    ]


def _invert_window_type_index(window_data: Mapping[NodeType, Set[NodeId]]) -> Dict[NodeId, NodeType]:
    id_to_type: Dict[NodeId, NodeType] = {}
    for node_type, node_ids in window_data.items():
        for node_id in node_ids:
            id_to_type[node_id] = node_type
    return id_to_type


def _vertex_covers_query_types(
    node_ids: Sequence[NodeId],
    id_to_type: Mapping[NodeId, NodeType],
    required_counts: Counter[NodeType],
) -> bool:
    local_counts: Counter[NodeType] = Counter()
    for node_id in node_ids:
        node_type = id_to_type.get(node_id)
        if node_type is not None:
            local_counts[node_type] += 1
    return all(local_counts[node_type] >= required for node_type, required in required_counts.items())


def iter_bindings_in_vertex(
    node_ids: Sequence[NodeId],
    id_to_type: Mapping[NodeId, NodeType],
    query_object_types: Sequence[NodeType],
) -> Iterator[Tuple[NodeId, ...]]:
    sorted_ids = sorted(node_ids)
    role_candidates = [
        [node_id for node_id in sorted_ids if id_to_type.get(node_id) == query_type]
        for query_type in query_object_types
    ]
    if any(len(candidates) == 0 for candidates in role_candidates):
        return

    used: Set[NodeId] = set()
    current: list[NodeId] = []

    def backtrack(position: int) -> Iterator[Tuple[NodeId, ...]]:
        if position == len(role_candidates):
            yield tuple(current)
            return

        for node_id in role_candidates[position]:
            if node_id in used:
                continue
            used.add(node_id)
            current.append(node_id)
            yield from backtrack(position + 1)
            current.pop()
            used.remove(node_id)

    yield from backtrack(0)


def _merge_theta_d_maps(
    out: MutableMapping[Tuple[Theta, D], Set[FrameId]],
    source: Optional[Mapping[Tuple[Theta, D], Set[FrameId]]],
) -> None:
    if not source:
        return
    for theta_d, frames in source.items():
        if theta_d not in out:
            out[theta_d] = set(frames)
        else:
            out[theta_d].update(frames)


class PredecessorRelationResolver:
    def __init__(self, graph: ig.Graph) -> None:
        self.graph = graph
        self._cache: Dict[Tuple[int, Tuple[NodeId, NodeId]], Dict[Tuple[Theta, D], Set[FrameId]]] = {}

    def collect_pair_index_from_predecessors(
        self,
        vertex_idx: int,
        pair: Tuple[NodeId, NodeId],
    ) -> Dict[Tuple[Theta, D], Set[FrameId]]:
        cache_key = (vertex_idx, pair)
        if cache_key in self._cache:
            return self._cache[cache_key]

        merged: Dict[Tuple[Theta, D], Set[FrameId]] = {}
        self._collect_from_predecessors(vertex_idx, pair, merged, set())

        reverse_pair = (pair[1], pair[0])
        if reverse_pair != pair:
            self._collect_from_predecessors(vertex_idx, reverse_pair, merged, set())

        self._cache[cache_key] = merged
        return merged

    def _collect_from_predecessors(
        self,
        vertex_idx: int,
        pair: Tuple[NodeId, NodeId],
        out: MutableMapping[Tuple[Theta, D], Set[FrameId]],
        visited: Set[Tuple[int, Tuple[NodeId, NodeId]]],
    ) -> None:
        visit_key = (vertex_idx, pair)
        if visit_key in visited:
            return
        visited.add(visit_key)

        vertex = self.graph.vs[vertex_idx]
        combinations_map = vertex["combinations"] or {}
        _merge_theta_d_maps(out, combinations_map.get(pair))

        for parent_idx in self.graph.predecessors(vertex_idx):
            self._collect_from_predecessors(parent_idx, pair, out, visited)


def match_bindings_in_window(
    graph: ig.Graph,
    window_data: Mapping[NodeType, Set[NodeId]],
    query_object_types: Sequence[NodeType],
    theta_d_condition: Optional[Tuple[Theta, D]],
    max_bindings_per_window: int = 0,
    stats: Optional[PaperQueryStats] = None,
) -> Dict[Tuple[NodeId, ...], Set[FrameId]]:
    id_to_type = _invert_window_type_index(window_data)
    required_counts: Counter[NodeType] = Counter(query_object_types)

    candidate_vertices: list[Tuple[int, int, Tuple[NodeId, ...]]] = []
    for vertex_idx, vertex in enumerate(graph.vs):
        node_ids = tuple(vertex["name"])
        if len(node_ids) < len(query_object_types):
            continue
        if _vertex_covers_query_types(node_ids, id_to_type, required_counts):
            candidate_vertices.append((len(node_ids), vertex_idx, node_ids))

    candidate_vertices.sort(key=lambda item: (item[0], item[1]))
    if stats is not None:
        stats.candidate_vertices_total += len(candidate_vertices)

    # Query follows the thesis flow: locate candidate vertices first, then
    # match required pair relations on the current vertex and its predecessors.
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
            for pair in combinations(binding, 2):
                if stats is not None:
                    stats.pair_lookups += 1

                pair_index = resolver.collect_pair_index_from_predecessors(vertex_idx, pair)
                if not pair_index:
                    per_pair_frames = []
                    break

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


def run_paper_query(
    index: WindowTypeIndex,
    cp_graphs: Mapping[WindowId, ig.Graph],
    query_object_types: Sequence[NodeType],
    query_spatial: Optional[Tuple[Theta, D]],
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

        window_result = match_bindings_in_window(
            graph,
            index[window_id],
            query_object_types,
            query_spatial,
            max_bindings_per_window=max_bindings_per_window,
            stats=stats,
        )
        if window_result:
            results[window_id] = window_result

    tree_root = build_tree_from_results(results)
    top_paths = get_top_five_paths_with_threshold(tree_root, frame_threshold, topk)
    return stats, top_paths
