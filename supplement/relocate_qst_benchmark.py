from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from supplement.corrected_query_model import QueryResult, QuerySpec, exact_query, result_signature
from supplement.dataset_config import PREFERRED_VIDEO_ORDER, dataset_frame_size, dataset_storage_path
from supplement.generate_corrected_baseline_workload import RAW_COLUMNS, query_from_dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOAD = (
    REPO_ROOT / "supplement" / "out" / "corrected_baseline_20260712" / "workload.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "supplement" / "out" / "relocate_qst"


def load_region_tracking_table(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, names=RAW_COLUMNS, header=None, usecols=range(len(RAW_COLUMNS)))
    table = pd.DataFrame(
        {
            "frame": raw["frame"].astype(np.int64),
            "track_id": raw["track_id"].astype(np.int64),
            "class_id": raw["class_id"].astype(np.int64),
            "cx": raw["left"].astype(float) + raw["width"].astype(float) / 2.0,
            "cy": raw["top"].astype(float) + raw["height"].astype(float) / 2.0,
            "width": raw["width"].astype(float),
            "height": raw["height"].astype(float),
            "confidence": raw["confidence"].astype(float),
        }
    )
    return table.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def build_region_vector(rows: pd.DataFrame, frame_width: int, frame_height: int) -> np.ndarray:
    """Create a fixed region/track descriptor from precomputed detections."""
    if rows.empty:
        raise ValueError("cannot encode an empty region track")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    ordered = rows.sort_values("frame", kind="stable")
    cx = ordered["cx"].to_numpy(dtype=np.float64) / frame_width
    cy = ordered["cy"].to_numpy(dtype=np.float64) / frame_height
    width = ordered["width"].to_numpy(dtype=np.float64) / frame_width
    height = ordered["height"].to_numpy(dtype=np.float64) / frame_height
    confidence = ordered["confidence"].to_numpy(dtype=np.float64)
    aspect = width / np.maximum(height, 1e-9)
    motion_x = float(cx[-1] - cx[0])
    motion_y = float(cy[-1] - cy[0])
    duration = max(1, int(ordered["frame"].iloc[-1]) - int(ordered["frame"].iloc[0]) + 1)

    vector = np.asarray(
        [
            cx.mean(),
            cy.mean(),
            width.mean(),
            height.mean(),
            cx.std(),
            cy.std(),
            width.std(),
            height.std(),
            (width * height).mean(),
            np.clip(aspect.mean(), 0.0, 10.0) / 10.0,
            np.clip(confidence.mean(), 0.0, 1.0),
            motion_x,
            motion_y,
            math.log1p(duration) / 10.0,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("region descriptor is not finite")
    return vector / norm


@dataclass(frozen=True)
class TrackRegionArtifact:
    ids_by_class: Mapping[int, np.ndarray]
    vectors_by_class: Mapping[int, np.ndarray]
    frames_by_required_counts: Mapping[int, Mapping[int, int]]


class RelocateStyleStrategy:
    """Exemplar-to-region retrieval followed by exact multi-track QST verification."""

    artifact_kind = "query-independent-track-region-representations"

    def __init__(self, data: pd.DataFrame, frame_width: int, frame_height: int) -> None:
        required = {
            "frame",
            "track_id",
            "class_id",
            "cx",
            "cy",
            "width",
            "height",
            "confidence",
        }
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"detection table is missing columns: {sorted(missing)}")
        self.data = data
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.artifact = self._build_artifact()
        self.last_role_scores: list[tuple[np.ndarray, np.ndarray]] = []

    def _build_artifact(self) -> TrackRegionArtifact:
        ids: dict[int, list[int]] = {}
        vectors: dict[int, list[np.ndarray]] = {}
        for (class_id, track_id), rows in self.data.groupby(
            ["class_id", "track_id"], sort=True
        ):
            class_id = int(class_id)
            ids.setdefault(class_id, []).append(int(track_id))
            vectors.setdefault(class_id, []).append(
                build_region_vector(rows, self.frame_width, self.frame_height)
            )

        unique = self.data.drop_duplicates(["frame", "class_id", "track_id"])
        frame_counts: dict[int, dict[int, int]] = {}
        for (frame, class_id), rows in unique.groupby(["frame", "class_id"], sort=False):
            frame_counts.setdefault(int(frame), {})[int(class_id)] = len(rows)

        return TrackRegionArtifact(
            ids_by_class={key: np.asarray(value, dtype=np.int64) for key, value in ids.items()},
            vectors_by_class={key: np.vstack(value) for key, value in vectors.items()},
            frames_by_required_counts=frame_counts,
        )

    def _candidate_frames(self, role_types: Sequence[int]) -> set[int]:
        required = Counter(int(class_id) for class_id in role_types)
        return {
            frame
            for frame, counts in self.artifact.frames_by_required_counts.items()
            if all(int(counts.get(class_id, 0)) >= count for class_id, count in required.items())
        }

    def query(
        self,
        query: QuerySpec,
        binding: Sequence[int],
        start_frame: int,
        end_frame: int,
    ) -> list[QueryResult]:
        if len(binding) != len(query.role_types):
            raise ValueError("exemplar binding must contain one track per query role")
        if end_frame < start_frame:
            raise ValueError("exemplar frame interval is invalid")

        segment = self.data[
            self.data["frame"].between(int(start_frame), int(end_frame))
        ]
        self.last_role_scores = []
        for role_type, track_id in zip(query.role_types, binding):
            exemplar_rows = segment[
                (segment["track_id"] == int(track_id))
                & (segment["class_id"] == int(role_type))
            ]
            expected_frames = int(end_frame) - int(start_frame) + 1
            if exemplar_rows["frame"].nunique() != expected_frames:
                raise ValueError(
                    f"incomplete exemplar track {track_id} for class {role_type}: "
                    f"expected {expected_frames} frames"
                )
            query_vector = build_region_vector(
                exemplar_rows, self.frame_width, self.frame_height
            )
            candidate_ids = self.artifact.ids_by_class.get(int(role_type))
            candidate_vectors = self.artifact.vectors_by_class.get(int(role_type))
            if candidate_ids is None or candidate_vectors is None:
                return []
            scores = candidate_vectors @ query_vector
            order = np.argsort(-scores, kind="stable")
            self.last_role_scores.append((candidate_ids[order], scores[order]))

        core = self.data[["frame", "track_id", "class_id", "cx", "cy"]]
        return exact_query(
            core,
            query,
            self.frame_width,
            self.frame_height,
            candidate_frames=self._candidate_frames(query.role_types),
        )

    def release(self) -> None:
        self.data = None
        self.artifact = None
        self.last_role_scores = []


def _load_workload(path: Path) -> dict[str, list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in payload:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["query_id"]))
    return grouped


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    workload_path: str | Path = DEFAULT_WORKLOAD,
    output_dir: str | Path = DEFAULT_OUTPUT,
    datasets: Iterable[str] = PREFERRED_VIDEO_ORDER,
    repetitions: int = 5,
    warmups: int = 1,
) -> tuple[Path, Path]:
    if repetitions <= 0 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups non-negative")
    workload = _load_workload(Path(workload_path))
    destination = Path(output_dir)
    raw_rows: list[dict[str, object]] = []
    build_rows: list[dict[str, object]] = []

    for dataset in datasets:
        print(f"[load] {dataset}", flush=True)
        data = load_region_tracking_table(dataset_storage_path(dataset))
        width, height = dataset_frame_size(dataset)
        build_start = time.perf_counter()
        strategy = RelocateStyleStrategy(data, width, height)
        build_elapsed = time.perf_counter() - build_start
        build_rows.append(
            {
                "dataset": dataset,
                "offline_build_s": build_elapsed,
                "track_count": sum(len(value) for value in strategy.artifact.ids_by_class.values()),
                "representation_dim": 14,
            }
        )

        for row in workload[dataset]:
            query = query_from_dict(row)
            event = row["source_event"]
            binding = tuple(int(value) for value in event["binding"])
            start_frame = int(event["start_frame"])
            end_frame = int(event["end_frame"])
            expected = result_signature(
                exact_query(
                    data[["frame", "track_id", "class_id", "cx", "cy"]],
                    query,
                    width,
                    height,
                )
            )

            for _ in range(warmups):
                warmup = strategy.query(query, binding, start_frame, end_frame)
                if result_signature(warmup) != expected:
                    raise RuntimeError(f"signature mismatch in warm-up: {dataset} {query.query_id}")

            for repetition in range(1, repetitions + 1):
                started = time.perf_counter()
                results = strategy.query(query, binding, start_frame, end_frame)
                elapsed = time.perf_counter() - started
                signature = result_signature(results)
                if signature != expected:
                    raise RuntimeError(f"signature mismatch: {dataset} {query.query_id}")
                raw_rows.append(
                    {
                        "dataset": dataset,
                        "query_id": query.query_id,
                        "method": "RELOCATE-style",
                        "repetition": repetition,
                        "elapsed_s": elapsed,
                        "signature": signature,
                        "signature_verified": True,
                    }
                )
                print(
                    f"[timing] {dataset} {query.query_id} repetition={repetition} "
                    f"elapsed_s={elapsed:.6f}",
                    flush=True,
                )
        strategy.release()
        del strategy, data

    summary_rows: list[dict[str, object]] = []
    frame = pd.DataFrame(raw_rows)
    for (dataset, query_id), group in frame.groupby(["dataset", "query_id"], sort=False):
        values = group["elapsed_s"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "method": "RELOCATE-style",
                "median_s": float(np.median(values)),
                "q1_s": float(np.quantile(values, 0.25)),
                "q3_s": float(np.quantile(values, 0.75)),
                "source": "measured",
                "signature": str(group["signature"].iloc[0]),
                "signature_verified": True,
                "representation": "query-independent 14-D track-region descriptor",
            }
        )

    raw_path = destination / "relocate_qst_raw.csv"
    summary_path = destination / "relocate_qst_summary.csv"
    _write_csv(raw_path, raw_rows, list(raw_rows[0].keys()))
    _write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    _write_csv(destination / "relocate_qst_offline_build.csv", build_rows, list(build_rows[0].keys()))
    (destination / "run_metadata.json").write_text(
        json.dumps(
            {
                "method": "RELOCATE-style",
                "workload": str(Path(workload_path).resolve()),
                "online_timing_excludes": ["tracking file load", "track-region artifact construction"],
                "online_timing_includes": [
                    "query exemplar encoding",
                    "role-wise cosine retrieval",
                    "candidate-frame generation",
                    "exact QST verification",
                    "top-k ranking",
                ],
                "official_code_reference": str(Path(r"D:\pycharm\REN\visual_query")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adapted multi-object RELOCATE-style QST benchmark.")
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", default=",".join(PREFERRED_VIDEO_ORDER))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    run_benchmark(
        args.workload,
        args.output_dir,
        datasets=[value.strip() for value in args.datasets.split(",") if value.strip()],
        repetitions=args.repetitions,
        warmups=args.warmups,
    )


if __name__ == "__main__":
    main()
