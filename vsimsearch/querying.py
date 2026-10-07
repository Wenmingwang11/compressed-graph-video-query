from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Set, Tuple


WindowId = int
NodeType = int
NodeId = int
Theta = float
D = float
FrameId = int

# window_index:
#   window_id -> node_type -> set(node_id)
WindowTypeIndex = Mapping[WindowId, Mapping[NodeType, Set[NodeId]]]

# window_edge_index:
#   (sid, eid) -> (theta, d) -> set(frame_id)
WindowEdgeIndex = Mapping[Tuple[NodeId, NodeId], Mapping[Tuple[Theta, D], Set[FrameId]]]


@dataclass
class QueryStats:
    windows_total: int = 0
    windows_after_type_filter: int = 0
    combinations_total: int = 0
    combinations_matched: int = 0
    pair_lookups: int = 0
    theta_d_bins_scanned: int = 0


def filter_windows_by_types(
    index: WindowTypeIndex, query_object_types: Sequence[NodeType]
) -> Dict[WindowId, Dict[NodeType, Set[NodeId]]]:
    """
    Keep only windows that contain all required types.
    Returns: window_id -> {type -> set(node_ids)} restricted to query types.
    """
    required = set(query_object_types)
    filtered: Dict[WindowId, Dict[NodeType, Set[NodeId]]] = {}
    for window_id, window_data in index.items():
        if required.issubset(window_data.keys()):
            filtered[window_id] = {t: set(window_data[t]) for t in query_object_types}
    return filtered


def iter_node_combinations(
    window_data: Mapping[NodeType, Set[NodeId]], query_object_types: Sequence[NodeType]
) -> Iterator[Tuple[NodeId, ...]]:
    """
    Generate all node-id tuples that satisfy the required type list.
    Each position corresponds to the same position in query_object_types.
    """
    type_node_ids = [sorted(window_data.get(t, set())) for t in query_object_types]
    if any(len(ids) == 0 for ids in type_node_ids):
        return iter(())
    return product(*type_node_ids)


def _theta_d_satisfy(theta: Theta, d: D, condition: Optional[Tuple[Theta, D]]) -> bool:
    if condition is None:
        return True
    theta_lt, d_lt = condition
    return theta < theta_lt and d < d_lt


def match_combinations_in_window(
    window_edge_index: WindowEdgeIndex,
    selected_node_ids: Iterable[Tuple[NodeId, ...]],
    theta_d_condition: Optional[Tuple[Theta, D]],
    stats: Optional[QueryStats] = None,
) -> Dict[Tuple[NodeId, ...], Set[FrameId]]:
    """
    For each node-id tuple, check all node pairs (combinations(., 2)) and
    intersect matched frame sets. Returns node-id tuple -> matched frames.
    """
    result: Dict[Tuple[NodeId, ...], Set[FrameId]] = {}
    for id_tuple in selected_node_ids:
        if stats is not None:
            stats.combinations_total += 1

        per_pair_frame_sets: list[Set[FrameId]] = []
        for pair in combinations(id_tuple, 2):
            if stats is not None:
                stats.pair_lookups += 1

            pair_index = window_edge_index.get(pair)
            if not pair_index:
                continue

            # collect all theta/d bins satisfying the condition
            matched_frames: Set[FrameId] = set()
            for (theta, d), frame_set in pair_index.items():
                if stats is not None:
                    stats.theta_d_bins_scanned += 1
                if _theta_d_satisfy(theta, d, theta_d_condition):
                    matched_frames |= set(frame_set)

            if matched_frames:
                per_pair_frame_sets.append(matched_frames)

        if not per_pair_frame_sets:
            continue

        intersection = set.intersection(*per_pair_frame_sets)
        if intersection:
            result[id_tuple] = intersection
            if stats is not None:
                stats.combinations_matched += 1
    return result
