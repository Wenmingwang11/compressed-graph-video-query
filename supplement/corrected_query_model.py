from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class RelationConstraint:
    d_min: int
    d_max: int
    theta_bins: frozenset[int]

    def __post_init__(self) -> None:
        if self.d_min < 0 or self.d_max < self.d_min:
            raise ValueError("distance interval must satisfy 0 <= d_min <= d_max")
        if not self.theta_bins:
            raise ValueError("theta_bins must not be empty")

    def matches(self, theta_bin: int, distance_bin: int) -> bool:
        return self.d_min <= distance_bin <= self.d_max and theta_bin in self.theta_bins


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    role_types: tuple[int, ...]
    relations: Mapping[tuple[int, int], RelationConstraint]
    min_consecutive_frames: int
    topk: int
    theta_parts: int = 10
    distance_parts: int = 8

    def __post_init__(self) -> None:
        if not self.role_types:
            raise ValueError("role_types must not be empty")
        if self.min_consecutive_frames <= 0:
            raise ValueError("min_consecutive_frames must be positive")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.theta_parts <= 0 or self.distance_parts <= 0:
            raise ValueError("quantization part counts must be positive")
        role_count = len(self.role_types)
        for source_role, target_role in self.relations:
            if source_role == target_role:
                raise ValueError("a relation must connect two different roles")
            if not (0 <= source_role < role_count and 0 <= target_role < role_count):
                raise ValueError("relation role index is outside role_types")


@dataclass(frozen=True)
class QueryResult:
    binding: tuple[int, ...]
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def quantize_relation(
    source_xy: tuple[float, float],
    target_xy: tuple[float, float],
    frame_width: int,
    frame_height: int,
    theta_parts: int = 10,
    distance_parts: int = 8,
) -> tuple[int, int]:
    """Return the paper's directed angle bin and normalized distance bin."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if theta_parts <= 0 or distance_parts <= 0:
        raise ValueError("quantization part counts must be positive")

    dx = float(target_xy[0]) - float(source_xy[0])
    dy = float(target_xy[1]) - float(source_xy[1])
    theta = math.atan2(dy, dx)
    theta_bin = math.floor(theta / (math.pi / theta_parts))

    normalized_distance = math.hypot(dx, dy) / float(frame_width + frame_height)
    distance_bin = math.floor(normalized_distance * distance_parts)
    return theta_bin, distance_bin


def maximal_consecutive_segments(frames: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted({int(frame) for frame in frames})
    if not ordered:
        return []

    segments: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            segments.append((start, previous))
            start = frame
        previous = frame
    segments.append((start, previous))
    return segments


def _validate_detection_table(data: pd.DataFrame) -> None:
    required = {"frame", "track_id", "class_id", "cx", "cy"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"detection table is missing columns: {sorted(missing)}")


def _frame_bindings(
    frame_data: pd.DataFrame,
    role_types: Sequence[int],
    allowed_bindings: set[tuple[int, ...]] | None,
    stats: dict | None = None,
    binding_frames: Mapping[tuple[int, ...], set[int]] | None = None,
    frame: int | None = None,
) -> Iterable[tuple[tuple[int, ...], dict[int, tuple[float, float]]]]:
    by_type: dict[int, list[int]] = {}
    positions: dict[int, tuple[float, float]] = {}
    for row in frame_data.itertuples(index=False):
        track_id = int(row.track_id)
        class_id = int(row.class_id)
        by_type.setdefault(class_id, []).append(track_id)
        positions[track_id] = (float(row.cx), float(row.cy))

    role_candidates = [sorted(set(by_type.get(int(role_type), []))) for role_type in role_types]
    if any(not candidates for candidates in role_candidates):
        return

    for binding in product(*role_candidates):
        binding_tuple = tuple(int(track_id) for track_id in binding)
        if len(set(binding_tuple)) != len(binding_tuple):
            continue
        if stats is not None:
            stats['binding_frame_enumerated'] += 1
        if allowed_bindings is not None and binding_tuple not in allowed_bindings:
            continue
        if binding_frames is not None and frame not in binding_frames.get(binding_tuple, ()):
            continue
        if stats is not None:
            stats['binding_frame_admitted'] += 1
        yield binding_tuple, positions


def exact_query(
    data: pd.DataFrame,
    query: QuerySpec,
    frame_width: int,
    frame_height: int,
    candidate_bindings: Iterable[Sequence[int]] | None = None,
    candidate_frames: Iterable[int] | None = None,
    *,
    stats: dict | None = None,
    return_all: bool = False,
    candidate_binding_frames: Mapping[tuple[int, ...], Iterable[int]] | None = None,
) -> list[QueryResult]:
    """Evaluate a QST query exactly over tracked detections."""
    _validate_detection_table(data)
    started = time.perf_counter() if stats is not None else 0.0
    if stats is not None:
        for key in ('candidate_frames', 'binding_frame_enumerated', 'binding_frame_admitted',
                    'spatial_predicate_evaluations', 'spatial_match_binding_frames'):
            stats[key] = 0
    allowed = None
    if candidate_bindings is not None:
        allowed = {tuple(int(track_id) for track_id in binding) for binding in candidate_bindings}

    selected = data
    binding_frames = None
    if candidate_binding_frames is not None:
        binding_frames = {tuple(map(int,b)): set(map(int,frames)) for b,frames in candidate_binding_frames.items()}
        proposal_frames = set().union(*binding_frames.values()) if binding_frames else set()
        selected = selected[selected['frame'].isin(proposal_frames)]
    if candidate_frames is not None:
        allowed_frames = {int(frame) for frame in candidate_frames}
        selected = selected[selected["frame"].isin(allowed_frames)]

    matching_frames: dict[tuple[int, ...], set[int]] = {}
    ordered = selected.sort_values(["frame", "track_id"], kind="stable")
    for frame, frame_data in ordered.groupby("frame", sort=True):
        if stats is not None:
            stats['candidate_frames'] += 1
        for binding, positions in _frame_bindings(frame_data, query.role_types, allowed, stats, binding_frames, int(frame)):
            satisfies = True
            for (source_role, target_role), constraint in query.relations.items():
                if stats is not None:
                    stats['spatial_predicate_evaluations'] += 1
                theta_bin, distance_bin = quantize_relation(
                    positions[binding[source_role]],
                    positions[binding[target_role]],
                    frame_width,
                    frame_height,
                    query.theta_parts,
                    query.distance_parts,
                )
                if not constraint.matches(theta_bin, distance_bin):
                    satisfies = False
                    break
            if satisfies:
                matching_frames.setdefault(binding, set()).add(int(frame))
                if stats is not None:
                    stats['spatial_match_binding_frames'] += 1

    scan_finished = time.perf_counter() if stats is not None else 0.0
    results: list[QueryResult] = []
    for binding, frames in matching_frames.items():
        for start_frame, end_frame in maximal_consecutive_segments(frames):
            if end_frame - start_frame + 1 >= query.min_consecutive_frames:
                results.append(QueryResult(binding, start_frame, end_frame))

    merge_finished = time.perf_counter() if stats is not None else 0.0
    results.sort(
        key=lambda result: (
            -result.duration_frames,
            result.binding,
            result.start_frame,
            result.end_frame,
        )
    )
    returned = results if return_all else results[: query.topk]
    if stats is not None:
        finished = time.perf_counter()
        stats.update(qualifying_intervals=len(results), spatial_scan_s=scan_finished-started,
                     temporal_merge_s=merge_finished-scan_finished,
                     topk_s=finished-merge_finished, verification_s=finished-started)
    return returned


def ordered_result_signature(results: Iterable[QueryResult]) -> str:
    """Unlike result_signature, also verify deterministic Top-k ordering."""
    payload = [[list(result.binding), int(result.start_frame), int(result.end_frame)] for result in results]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode('ascii')).hexdigest()


def result_signature(results: Iterable[QueryResult]) -> str:
    canonical = sorted(
        (
            [int(track_id) for track_id in result.binding],
            int(result.start_frame),
            int(result.end_frame),
        )
        for result in results
    )
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
