from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from supplement.corrected_query_model import QueryResult, QuerySpec, exact_query, quantize_relation


STRATEGY_NAMES = (
    "vqpy",
    "eva",
    "vocal-udf",
    "video-colbert",
    "seiden",
    "lava",
    "star",
)

_REQUIRED_COLUMNS = {"frame", "track_id", "class_id", "cx", "cy"}


def _required_counts(role_types: Sequence[int]) -> Counter[int]:
    return Counter(int(class_id) for class_id in role_types)


def _counts_cover(counts: Mapping[int, int], required: Mapping[int, int]) -> bool:
    return all(int(counts.get(class_id, 0)) >= count for class_id, count in required.items())


def _frame_tracks(data: pd.DataFrame) -> dict[int, dict[int, tuple[int, ...]]]:
    result: dict[int, dict[int, tuple[int, ...]]] = {}
    for frame, frame_data in data.groupby("frame", sort=True):
        by_class: dict[int, set[int]] = {}
        for row in frame_data.itertuples(index=False):
            by_class.setdefault(int(row.class_id), set()).add(int(row.track_id))
        result[int(frame)] = {
            class_id: tuple(sorted(track_ids)) for class_id, track_ids in by_class.items()
        }
    return result


def _frame_counts(catalog: Mapping[int, Mapping[int, Sequence[int]]]) -> dict[int, Counter[int]]:
    return {
        int(frame): Counter({int(class_id): len(track_ids) for class_id, track_ids in classes.items()})
        for frame, classes in catalog.items()
    }


def _bindings_for_frame(
    classes: Mapping[int, Sequence[int]], role_types: Sequence[int]
) -> Iterable[tuple[int, ...]]:
    candidates = [classes.get(int(class_id), ()) for class_id in role_types]
    if any(not role_candidates for role_candidates in candidates):
        return
    for binding in product(*candidates):
        normalized = tuple(int(track_id) for track_id in binding)
        if len(set(normalized)) == len(normalized):
            yield normalized


@dataclass(frozen=True)
class ProceduralFrameArtifact:
    frames: Mapping[int, tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class RelationTableArtifact:
    detections: pd.DataFrame
    tables_by_class: Mapping[int, pd.DataFrame]


@dataclass(frozen=True)
class PredicateRegistryArtifact:
    predicates: Mapping[int, Mapping[int, tuple[int, ...]]]


@dataclass(frozen=True)
class SegmentTokenArtifact:
    segment_size: int
    token_postings: Mapping[int, frozenset[int]]
    segment_frames: Mapping[int, tuple[int, ...]]
    segment_count_bounds: Mapping[int, Mapping[int, int]]


@dataclass(frozen=True)
class ChunkCountArtifact:
    chunk_size: int
    chunk_frames: Mapping[int, tuple[int, ...]]
    category_count_bounds: Mapping[int, Mapping[int, int]]


@dataclass(frozen=True)
class HierarchicalSegmentNode:
    frames: tuple[int, ...]
    category_count_bounds: Mapping[int, int]
    children: tuple["HierarchicalSegmentNode", ...] = ()


@dataclass(frozen=True)
class HierarchicalSegmentArtifact:
    leaf_size: int
    roots: tuple[HierarchicalSegmentNode, ...]


@dataclass(frozen=True)
class DirectedEdgeArtifact:
    edge_postings: Mapping[tuple[int, int], np.ndarray]
    type_postings: Mapping[int, np.ndarray]


class CorrectedBaselineStrategy:
    artifact_kind = "abstract"

    def __init__(self, data: pd.DataFrame, frame_width: int, frame_height: int) -> None:
        missing = _REQUIRED_COLUMNS.difference(data.columns)
        if missing:
            raise ValueError(f"detection table is missing columns: {sorted(missing)}")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        # Callers own this table and must treat it as read-only while a strategy is alive.
        self._data: pd.DataFrame | None = data
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.artifact: object | None = self._build_artifact()

    def _build_artifact(self) -> object:
        raise NotImplementedError

    def _candidates(
        self, query: QuerySpec
    ) -> tuple[Iterable[Sequence[int]] | None, Iterable[int] | None]:
        raise NotImplementedError

    def query(self, query: QuerySpec, *, stats: dict | None = None,
              return_all: bool = False) -> list[QueryResult]:
        if self._data is None or self.artifact is None:
            raise RuntimeError("corrected baseline strategy has been released")
        started = time.perf_counter() if stats is not None else 0.0
        candidate_bindings, candidate_frames = self._candidates(query)
        if stats is not None:
            stats['candidate_s'] = time.perf_counter() - started
        return exact_query(
            self._data,
            query,
            self.frame_width,
            self.frame_height,
            candidate_bindings=candidate_bindings,
            candidate_frames=candidate_frames,
            stats=stats,
            return_all=return_all,
        )

    def release(self) -> None:
        """Drop references to the source table and the strategy-specific artifact."""
        self.artifact = None
        self._data = None


class VQPyStrategy(CorrectedBaselineStrategy):
    artifact_kind = "procedural-frame-enumerator"

    def _build_artifact(self) -> ProceduralFrameArtifact:
        frames: dict[int, tuple[tuple[int, int], ...]] = {}
        for frame, frame_data in self._data.groupby("frame", sort=True):
            frames[int(frame)] = tuple(
                sorted(
                    {
                        (int(row.track_id), int(row.class_id))
                        for row in frame_data.itertuples(index=False)
                    }
                )
            )
        return ProceduralFrameArtifact(frames)

    def _candidates(self, query: QuerySpec):
        required = _required_counts(query.role_types)
        frames: set[int] = set()
        for frame, records in self.artifact.frames.items():
            classes: dict[int, list[int]] = {}
            for track_id, class_id in records:
                classes.setdefault(class_id, []).append(track_id)
            counts = {class_id: len(track_ids) for class_id, track_ids in classes.items()}
            if not _counts_cover(counts, required):
                continue
            frames.add(frame)
        # Every class-compatible binding is enumerated by the exact verifier.
        # Materializing the same global Cartesian products here adds no pruning.
        return None, frames


class EVAStrategy(CorrectedBaselineStrategy):
    artifact_kind = "relation-table-join"

    def _build_artifact(self) -> RelationTableArtifact:
        detections = (
            self._data[["frame", "track_id", "class_id"]]
            .drop_duplicates()
            .astype({"frame": int, "track_id": int, "class_id": int})
            .sort_values(["frame", "track_id"], kind="stable")
            .reset_index(drop=True)
        )
        tables = {
            int(class_id): table.reset_index(drop=True)
            for class_id, table in detections.groupby("class_id", sort=True)
        }
        return RelationTableArtifact(detections, tables)

    def _candidates(self, query: QuerySpec):
        joined: pd.DataFrame | None = None
        role_columns: list[str] = []
        for role, class_id in enumerate(query.role_types):
            column = f"role_{role}"
            role_columns.append(column)
            table = self.artifact.tables_by_class.get(int(class_id))
            if table is None:
                return set(), set()
            role_table = table[["frame", "track_id"]].rename(columns={"track_id": column})
            joined = role_table if joined is None else joined.merge(role_table, on="frame", how="inner")
        if joined is None or joined.empty:
            return set(), set()
        injective = joined[role_columns].nunique(axis=1) == len(role_columns)
        joined = joined.loc[injective]
        return (
            {tuple(int(value) for value in row) for row in joined[role_columns].itertuples(index=False, name=None)},
            {int(frame) for frame in joined["frame"]},
        )


class VocalUDFStrategy(CorrectedBaselineStrategy):
    artifact_kind = "registered-predicate-composition"

    def _build_artifact(self) -> PredicateRegistryArtifact:
        registry: dict[int, dict[int, set[int]]] = {}
        for row in self._data.itertuples(index=False):
            frame = int(row.frame)
            class_id = int(row.class_id)
            registry.setdefault(class_id, {}).setdefault(frame, set()).add(int(row.track_id))
        predicates: dict[int, dict[int, tuple[int, ...]]] = {}
        while registry:
            class_id, frame_postings = registry.popitem()
            predicates[class_id] = {
                frame: tuple(sorted(track_ids))
                for frame, track_ids in sorted(frame_postings.items())
            }
        return PredicateRegistryArtifact(predicates)

    def _candidates(self, query: QuerySpec):
        postings = [
            self.artifact.predicates.get(int(class_id), {}) for class_id in query.role_types
        ]
        if any(not role_frames for role_frames in postings):
            return set(), set()
        frames = set(postings[0])
        for role_frames in postings[1:]:
            frames.intersection_update(role_frames)
        required = _required_counts(query.role_types)
        valid_frames = {
            frame for frame in frames
            if all(len(self.artifact.predicates[class_id][frame]) >= count
                   for class_id, count in required.items())
        }
        return None, valid_frames


def _bounded_segments(
    catalog: Mapping[int, Mapping[int, Sequence[int]]], segment_size: int
) -> tuple[dict[int, tuple[int, ...]], dict[int, dict[int, int]]]:
    segment_frames: dict[int, list[int]] = {}
    for frame in catalog:
        segment_frames.setdefault(int(frame) // segment_size, []).append(int(frame))
    frame_count_map = _frame_counts(catalog)
    bounds: dict[int, dict[int, int]] = {}
    for segment, frames in segment_frames.items():
        category_bounds: dict[int, int] = {}
        for frame in frames:
            for class_id, count in frame_count_map[frame].items():
                category_bounds[class_id] = max(category_bounds.get(class_id, 0), count)
        bounds[segment] = category_bounds
    return (
        {segment: tuple(sorted(frames)) for segment, frames in segment_frames.items()},
        bounds,
    )


class VideoColBERTStrategy(CorrectedBaselineStrategy):
    artifact_kind = "segment-token-inverted-index"
    segment_size = 32

    def _build_artifact(self) -> SegmentTokenArtifact:
        catalog = _frame_tracks(self._data)
        segment_frames, bounds = _bounded_segments(catalog, self.segment_size)
        postings: dict[int, set[int]] = {}
        for segment, category_bounds in bounds.items():
            for class_id in category_bounds:
                postings.setdefault(class_id, set()).add(segment)
        return SegmentTokenArtifact(
            self.segment_size,
            {token: frozenset(segments) for token, segments in postings.items()},
            segment_frames,
            bounds,
        )

    def _candidates(self, query: QuerySpec):
        required = _required_counts(query.role_types)
        segments: set[int] | None = None
        for class_id in required:
            token_segments = set(self.artifact.token_postings.get(class_id, frozenset()))
            segments = token_segments if segments is None else segments.intersection(token_segments)
        selected = {
            segment
            for segment in (segments or set())
            if _counts_cover(self.artifact.segment_count_bounds[segment], required)
        }
        return None, {
            frame for segment in selected for frame in self.artifact.segment_frames[segment]
        }


class SeidenStrategy(CorrectedBaselineStrategy):
    artifact_kind = "fixed-chunk-count-upper-bounds"
    chunk_size = 128

    def _build_artifact(self) -> ChunkCountArtifact:
        chunk_frames, bounds = _bounded_segments(_frame_tracks(self._data), self.chunk_size)
        return ChunkCountArtifact(self.chunk_size, chunk_frames, bounds)

    def _candidates(self, query: QuerySpec):
        required = _required_counts(query.role_types)
        chunks = [
            chunk
            for chunk, bounds in self.artifact.category_count_bounds.items()
            if _counts_cover(bounds, required)
        ]
        return None, {
            frame for chunk in chunks for frame in self.artifact.chunk_frames[chunk]
        }


def _merge_bounds(nodes: Sequence[HierarchicalSegmentNode]) -> dict[int, int]:
    bounds: dict[int, int] = {}
    for node in nodes:
        for class_id, count in node.category_count_bounds.items():
            bounds[class_id] = max(bounds.get(class_id, 0), int(count))
    return bounds


class LAVAStrategy(CorrectedBaselineStrategy):
    artifact_kind = "hierarchical-segment-summary"
    leaf_size = 16

    def _build_artifact(self) -> HierarchicalSegmentArtifact:
        catalog = _frame_tracks(self._data)
        leaf_frames, leaf_bounds = _bounded_segments(catalog, self.leaf_size)
        level = [
            HierarchicalSegmentNode(leaf_frames[key], leaf_bounds[key])
            for key in sorted(leaf_frames)
        ]
        if not level:
            return HierarchicalSegmentArtifact(self.leaf_size, ())
        while len(level) > 1:
            next_level: list[HierarchicalSegmentNode] = []
            for index in range(0, len(level), 2):
                children = tuple(level[index : index + 2])
                next_level.append(
                    HierarchicalSegmentNode(
                        tuple(frame for child in children for frame in child.frames),
                        _merge_bounds(children),
                        children,
                    )
                )
            level = next_level
        return HierarchicalSegmentArtifact(self.leaf_size, tuple(level))

    def _candidates(self, query: QuerySpec):
        required = _required_counts(query.role_types)
        frames: set[int] = set()

        def visit(node: HierarchicalSegmentNode) -> None:
            if not _counts_cover(node.category_count_bounds, required):
                return
            if node.children:
                for child in node.children:
                    visit(child)
            else:
                frames.update(node.frames)

        for root in self.artifact.roots:
            visit(root)
        return None, frames


class STARStrategy(CorrectedBaselineStrategy):
    artifact_kind = "directed-vector-edge-inverted-index"
    theta_parts = 10
    distance_parts = 8

    def __init__(
        self,
        data: pd.DataFrame,
        frame_width: int,
        frame_height: int,
        allowed_type_pairs: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        self._allowed_type_pairs = (
            None
            if allowed_type_pairs is None
            else {(int(source), int(target)) for source, target in allowed_type_pairs}
        )
        super().__init__(data, frame_width, frame_height)

    @staticmethod
    def _rows_by_class(frame_data: pd.DataFrame) -> dict[int, np.ndarray]:
        rows_by_class: dict[int, np.ndarray] = {}
        unique = frame_data.drop_duplicates(["class_id", "track_id"], keep="last")
        for class_id, class_rows in unique.groupby("class_id", sort=False):
            rows_by_class[int(class_id)] = class_rows[["track_id", "cx", "cy"]].to_numpy(
                dtype=np.float64, copy=True
            )
        return rows_by_class

    @staticmethod
    def _integer_dtype(data: pd.DataFrame) -> np.dtype:
        info = np.iinfo(np.int32)
        if data.empty:
            return np.dtype(np.int32)
        minimum = min(data["frame"].min(), data["track_id"].min())
        maximum = max(data["frame"].max(), data["track_id"].max())
        if minimum >= info.min and maximum <= info.max:
            return np.dtype(np.int32)
        return np.dtype(np.int64)

    def _build_artifact(self) -> DirectedEdgeArtifact:
        integer_dtype = self._integer_dtype(self._data)
        edge_dtype = np.dtype(
            [
                ("frame", integer_dtype),
                ("sid", integer_dtype),
                ("tid", integer_dtype),
                ("theta", np.int8),
                ("d", np.uint16),
            ],
            align=False,
        )
        type_dtype = np.dtype([("frame", integer_dtype), ("count", np.uint16)])
        pair_counts: Counter[tuple[int, int]] = Counter()
        type_rows: dict[int, list[tuple[int, int]]] = {}

        for frame, frame_data in self._data.groupby("frame", sort=True):
            rows_by_class = self._rows_by_class(frame_data)
            for class_id, rows in rows_by_class.items():
                type_rows.setdefault(class_id, []).append((int(frame), len(rows)))
            for source_class, source_rows in rows_by_class.items():
                source_ids = source_rows[:, 0].astype(np.int64, copy=False)
                for target_class, target_rows in rows_by_class.items():
                    if (
                        self._allowed_type_pairs is not None
                        and (source_class, target_class) not in self._allowed_type_pairs
                    ):
                        continue
                    target_ids = target_rows[:, 0].astype(np.int64, copy=False)
                    pair_counts[(source_class, target_class)] += int(
                        np.count_nonzero(source_ids[:, None] != target_ids[None, :])
                    )

        edge_postings = {
            class_pair: np.empty(count, dtype=edge_dtype)
            for class_pair, count in pair_counts.items()
            if count
        }
        offsets: Counter[tuple[int, int]] = Counter()
        scale = float(self.frame_width + self.frame_height)
        theta_width = np.pi / self.theta_parts

        for frame, frame_data in self._data.groupby("frame", sort=True):
            rows_by_class = self._rows_by_class(frame_data)
            for source_class, source_rows in rows_by_class.items():
                source_ids = source_rows[:, 0].astype(np.int64, copy=False)
                for target_class, target_rows in rows_by_class.items():
                    class_pair = (source_class, target_class)
                    posting = edge_postings.get(class_pair)
                    if posting is None:
                        continue
                    target_ids = target_rows[:, 0].astype(np.int64, copy=False)
                    source_indexes, target_indexes = np.nonzero(
                        source_ids[:, None] != target_ids[None, :]
                    )
                    count = len(source_indexes)
                    if not count:
                        continue
                    start = offsets[class_pair]
                    stop = start + count
                    block = posting[start:stop]
                    dx = target_rows[target_indexes, 1] - source_rows[source_indexes, 1]
                    dy = target_rows[target_indexes, 2] - source_rows[source_indexes, 2]
                    block["frame"] = int(frame)
                    block["sid"] = source_ids[source_indexes]
                    block["tid"] = target_ids[target_indexes]
                    block["theta"] = np.floor(np.arctan2(dy, dx) / theta_width).astype(
                        np.int8
                    )
                    block["d"] = np.floor(
                        np.hypot(dx, dy) / scale * self.distance_parts
                    ).astype(np.uint16)
                    offsets[class_pair] = stop

        type_postings = {
            class_id: np.array(rows, dtype=type_dtype) for class_id, rows in type_rows.items()
        }
        return DirectedEdgeArtifact(edge_postings, type_postings)

    def _type_frames(self, required: Mapping[int, int]) -> set[int]:
        candidates: set[int] | None = None
        for class_id, count in required.items():
            posting = self.artifact.type_postings.get(class_id)
            if posting is None:
                return set()
            frames = set(posting["frame"][posting["count"] >= count].astype(int).tolist())
            candidates = frames if candidates is None else candidates.intersection(frames)
        return candidates or set()

    def _candidates(self, query: QuerySpec):
        required = _required_counts(query.role_types)
        type_frames = self._type_frames(required)
        if (
            query.theta_parts != self.theta_parts
            or query.distance_parts != self.distance_parts
        ):
            return None, type_frames

        candidate_frames = type_frames
        for (source_role, target_role), constraint in query.relations.items():
            class_pair = (
                int(query.role_types[source_role]),
                int(query.role_types[target_role]),
            )
            # Absent from a workload-restricted artifact means unknown, not false.
            if self._allowed_type_pairs is not None and class_pair not in self._allowed_type_pairs:
                continue
            posting = self.artifact.edge_postings.get(class_pair)
            if posting is None:
                return None, set()
            relation_mask = (
                (posting["d"] >= constraint.d_min)
                & (posting["d"] <= constraint.d_max)
                & np.isin(posting["theta"], tuple(constraint.theta_bins))
            )
            relation_frames = set(posting["frame"][relation_mask].astype(int).tolist())
            candidate_frames = candidate_frames.intersection(relation_frames)
        return None, candidate_frames


_STRATEGIES = {
    "vqpy": VQPyStrategy,
    "eva": EVAStrategy,
    "vocal-udf": VocalUDFStrategy,
    "video-colbert": VideoColBERTStrategy,
    "seiden": SeidenStrategy,
    "lava": LAVAStrategy,
    "star": STARStrategy,
}


def _normalize_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    if normalized.endswith("-style"):
        normalized = normalized[: -len("-style")]
    aliases = {"vocaludf": "vocal-udf", "videocolbert": "video-colbert"}
    return aliases.get(normalized, normalized)


def build_strategy(
    name: str,
    data: pd.DataFrame,
    width: int,
    height: int,
    allowed_type_pairs: Iterable[tuple[int, int]] | None = None,
) -> CorrectedBaselineStrategy:
    """Build a query-independent adapted-baseline artifact and its exact verifier."""
    normalized = _normalize_name(name)
    strategy_class = _STRATEGIES.get(normalized)
    if strategy_class is None:
        raise ValueError(
            f"unknown corrected baseline strategy {name!r}; expected one of {STRATEGY_NAMES}"
        )
    if strategy_class is STARStrategy:
        return strategy_class(
            data,
            width,
            height,
            allowed_type_pairs=allowed_type_pairs,
        )
    return strategy_class(data, width, height)


__all__ = ["STRATEGY_NAMES", "CorrectedBaselineStrategy", "build_strategy"]
