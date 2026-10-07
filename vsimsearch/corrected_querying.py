from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

from supplement.corrected_query_model import QueryResult, QuerySpec, exact_query, quantize_relation


RelationKey = tuple[int, int, int, int]
TrackPair = tuple[int, int]
Interval = tuple[int, int]


@dataclass(frozen=True)
class DirectedRelationIndex:
    frame_width: int
    frame_height: int
    theta_parts: int
    distance_parts: int
    postings: Mapping[RelationKey, Mapping[TrackPair, tuple[Interval, ...]]]


def build_directed_relation_index(
    data: pd.DataFrame,
    frame_width: int,
    frame_height: int,
    theta_parts: int = 10,
    distance_parts: int = 8,
    allowed_types: Iterable[int] | None = None,
    allowed_type_pairs: Iterable[tuple[int, int]] | None = None,
) -> DirectedRelationIndex:
    """Build a bidirected, run-compressed relation index over tracked objects."""
    required = {"frame", "track_id", "class_id", "cx", "cy"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"detection table is missing columns: {sorted(missing)}")
    allowed = None if allowed_types is None else {int(class_id) for class_id in allowed_types}
    allowed_pairs = (
        None
        if allowed_type_pairs is None
        else {(int(source_type), int(target_type)) for source_type, target_type in allowed_type_pairs}
    )
    if allowed is None and allowed_pairs is not None:
        allowed = {class_id for pair in allowed_pairs for class_id in pair}

    mutable: dict[RelationKey, dict[TrackPair, list[list[int]]]] = {}
    selected = data if allowed is None else data[data["class_id"].isin(allowed)]
    for frame, frame_data in selected.sort_values(["frame", "track_id"], kind="stable").groupby(
        "frame", sort=True
    ):
        frame_id = int(frame)
        objects = [
            (int(row.track_id), int(row.class_id), (float(row.cx), float(row.cy)))
            for row in frame_data.itertuples(index=False)
        ]
        for source_id, source_type, source_xy in objects:
            for target_id, target_type, target_xy in objects:
                if source_id == target_id:
                    continue
                if allowed_pairs is not None and (source_type, target_type) not in allowed_pairs:
                    continue
                theta_bin, distance_bin = quantize_relation(
                    source_xy,
                    target_xy,
                    frame_width,
                    frame_height,
                    theta_parts,
                    distance_parts,
                )
                key = (source_type, target_type, theta_bin, distance_bin)
                intervals = mutable.setdefault(key, {}).setdefault((source_id, target_id), [])
                if intervals and intervals[-1][1] == frame_id - 1:
                    intervals[-1][1] = frame_id
                else:
                    intervals.append([frame_id, frame_id])

    postings = {
        key: {
            pair: tuple((int(start), int(end)) for start, end in intervals)
            for pair, intervals in pair_map.items()
        }
        for key, pair_map in mutable.items()
    }
    return DirectedRelationIndex(
        int(frame_width),
        int(frame_height),
        int(theta_parts),
        int(distance_parts),
        postings,
    )


def _expand_intervals(intervals: Iterable[Interval]) -> set[int]:
    frames: set[int] = set()
    for start, end in intervals:
        frames.update(range(int(start), int(end) + 1))
    return frames


def _matching_pairs_and_frames(
    index: DirectedRelationIndex,
    source_type: int,
    target_type: int,
    theta_bins: Iterable[int],
    d_min: int,
    d_max: int,
) -> tuple[set[TrackPair], set[int]]:
    pairs: set[TrackPair] = set()
    frames: set[int] = set()
    for theta_bin in theta_bins:
        for distance_bin in range(d_min, d_max + 1):
            pair_map = index.postings.get(
                (int(source_type), int(target_type), int(theta_bin), int(distance_bin)),
                {},
            )
            pairs.update(pair_map)
            for intervals in pair_map.values():
                frames.update(_expand_intervals(intervals))
    return pairs, frames


def query_directed_relation_index(
    index: DirectedRelationIndex,
    data: pd.DataFrame,
    query: QuerySpec,
) -> list[QueryResult]:
    if (query.theta_parts, query.distance_parts) != (index.theta_parts, index.distance_parts):
        raise ValueError("query quantization does not match the relation index")
    if not query.relations:
        return exact_query(data, query, index.frame_width, index.frame_height)

    joined: pd.DataFrame | None = None
    candidate_frames: set[int] | None = None
    for (source_role, target_role), constraint in sorted(query.relations.items()):
        pairs, relation_frames = _matching_pairs_and_frames(
            index,
            query.role_types[source_role],
            query.role_types[target_role],
            constraint.theta_bins,
            constraint.d_min,
            constraint.d_max,
        )
        if not pairs or not relation_frames:
            return []
        pair_table = pd.DataFrame(
            sorted(pairs),
            columns=[f"role_{source_role}", f"role_{target_role}"],
        )
        if joined is None:
            joined = pair_table
        else:
            shared = [column for column in joined.columns if column in pair_table.columns]
            joined = joined.merge(pair_table, on=shared, how="inner").drop_duplicates()
        if joined.empty:
            return []
        candidate_frames = relation_frames if candidate_frames is None else candidate_frames & relation_frames
        if not candidate_frames:
            return []

    assert joined is not None and candidate_frames is not None
    role_columns = [f"role_{role}" for role in range(len(query.role_types))]
    if any(column not in joined.columns for column in role_columns):
        return exact_query(
            data,
            query,
            index.frame_width,
            index.frame_height,
            candidate_frames=candidate_frames,
        )
    joined = joined[joined.apply(lambda row: len(set(int(row[col]) for col in role_columns)) == len(role_columns), axis=1)]
    candidate_bindings = [tuple(int(row[col]) for col in role_columns) for _, row in joined.iterrows()]
    return exact_query(
        data,
        query,
        index.frame_width,
        index.frame_height,
        candidate_bindings=candidate_bindings,
        candidate_frames=candidate_frames,
    )
