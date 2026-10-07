from __future__ import annotations

import importlib
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd


VQPY_ROOT = Path("D:/pycharm/vqpy-main")
if str(VQPY_ROOT) not in sys.path:
    sys.path.insert(0, str(VQPY_ROOT))

base = importlib.import_module("offline_structured_query")

DEFAULT_STORAGE_ROOT = base.DEFAULT_STORAGE_ROOT
DEFAULT_DATASETS = base.DEFAULT_DATASETS
BindingMatch = base.BindingMatch
TimingBreakdown = base.TimingBreakdown
QueryRunResult = base.QueryRunResult
parse_pair_constraints = base.parse_pair_constraints
load_txt_dataset = base.load_txt_dataset
_resolve_dataset_path = base._resolve_dataset_path


def _required_counts(query_types: Sequence[int]) -> Counter[int]:
    return Counter(int(query_type) for query_type in query_types)


def frames_with_required_types(df: pd.DataFrame, query_types: Sequence[int]) -> set[int]:
    required = _required_counts(query_types)
    candidates: set[int] = set()
    for frame_id, frame_df in df.groupby("frame", sort=False):
        counts = Counter(int(class_id) for class_id in frame_df["class_id"])
        if all(counts.get(class_id, 0) >= count for class_id, count in required.items()):
            candidates.add(int(frame_id))
    return candidates


def chunk_candidate_frames(
    df: pd.DataFrame,
    query_types: Sequence[int],
    chunk_size: int = 128,
    sample_stride: int = 8,
    top_fraction: float | None = None,
) -> set[int]:
    required = _required_counts(query_types)
    frame_to_classes = {
        int(frame_id): list(map(int, frame_df["class_id"]))
        for frame_id, frame_df in df.groupby("frame", sort=True)
    }
    if not frame_to_classes:
        return set()

    chunks: dict[int, list[int]] = {}
    for frame_id in frame_to_classes:
        chunks.setdefault(frame_id // chunk_size, []).append(frame_id)

    scored_chunks: list[tuple[float, int, list[int]]] = []
    for chunk_id, frames in chunks.items():
        sample_frames = frames[:: max(1, sample_stride)] or frames[:1]
        hits = 0
        support = 0
        for frame_id in sample_frames:
            counts = Counter(frame_to_classes[frame_id])
            support += sum(min(counts.get(class_id, 0), count) for class_id, count in required.items())
            if all(counts.get(class_id, 0) >= count for class_id, count in required.items()):
                hits += 1
        score = hits + 0.01 * support
        scored_chunks.append((float(score), chunk_id, frames))

    positive = [item for item in scored_chunks if item[0] > 0.0]
    if not positive:
        return set(frame_to_classes)

    positive.sort(key=lambda item: (-item[0], item[1]))
    if top_fraction is not None:
        keep = max(1, int(len(positive) * top_fraction))
        positive = positive[:keep]

    candidate_frames: set[int] = set()
    for _, _, frames in positive:
        candidate_frames.update(frames)
    return candidate_frames


def query_text_from_spec(query_types: Sequence[int], pair_constraints: object, frame_threshold: int, topk: int) -> str:
    type_text = ", ".join(f"class {int(query_type)}" for query_type in query_types)
    return (
        f"find video clips containing objects of {type_text}, whose pairwise "
        f"quantized spatial relations satisfy {pair_constraints}, lasting more "
        f"than {frame_threshold} frames, and return top {topk} clips"
    )


def run_exact_scan_from_path(
    data_path: str | Path,
    query_types: Sequence[int],
    pair_constraints,
    frame_threshold: int,
    frame_width: int,
    frame_height: int,
    temporal_mode: str = "union_count",
    topk: int = 10,
    theta_n_parts: int = 10,
    theta_d_parts: int = 8,
):
    return base.run_query_from_path(
        data_path=data_path,
        query_types=query_types,
        pair_constraints=pair_constraints,
        frame_threshold=frame_threshold,
        frame_width=frame_width,
        frame_height=frame_height,
        temporal_mode=temporal_mode,
        topk=topk,
        theta_n_parts=theta_n_parts,
        theta_d_parts=theta_d_parts,
    )


def run_candidate_verified_query(
    data_path: str | Path,
    query_types: Sequence[int],
    pair_constraints,
    frame_threshold: int,
    frame_width: int,
    frame_height: int,
    temporal_mode: str = "union_count",
    topk: int = 10,
    theta_n_parts: int = 10,
    theta_d_parts: int = 8,
    candidate_mode: str = "type_frames",
):
    load_start = time.perf_counter()
    df = load_txt_dataset(data_path)
    load_s = time.perf_counter() - load_start

    candidate_start = time.perf_counter()
    if candidate_mode == "type_frames":
        candidate_frames = frames_with_required_types(df, query_types)
    elif candidate_mode == "seiden_chunks":
        candidate_frames = chunk_candidate_frames(df, query_types, chunk_size=128, sample_stride=8, top_fraction=None)
    elif candidate_mode == "lava_segments":
        candidate_frames = chunk_candidate_frames(df, query_types, chunk_size=256, sample_stride=12, top_fraction=None)
    elif candidate_mode == "video_colbert_summaries":
        candidate_frames = frames_with_required_types(df, query_types)
    else:
        raise ValueError(f"unsupported candidate_mode '{candidate_mode}'")
    candidate_s = time.perf_counter() - candidate_start

    filtered_df = df[df["frame"].isin(candidate_frames)].copy()
    result = base.run_query_dataframe(
        df=filtered_df,
        query_types=query_types,
        pair_constraints=pair_constraints,
        frame_threshold=frame_threshold,
        frame_width=frame_width,
        frame_height=frame_height,
        temporal_mode=temporal_mode,
        topk=topk,
        theta_n_parts=theta_n_parts,
        theta_d_parts=theta_d_parts,
    )
    result.timing.load_s = load_s
    result.timing.candidate_s = candidate_s
    result.timing.total_s += load_s + candidate_s
    return result
