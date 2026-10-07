from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from supplement.corrected_query_model import QuerySpec, RelationConstraint, quantize_relation
from supplement.dataset_config import (
    PREFERRED_VIDEO_ORDER,
    dataset_frame_size,
    dataset_storage_path,
)


RAW_COLUMNS = [
    "frame",
    "track_id",
    "left",
    "top",
    "width",
    "height",
    "confidence",
    "class_id",
    "x",
    "y",
    "z",
]


@dataclass(frozen=True)
class RealEvent:
    role_types: tuple[int, ...]
    binding: tuple[int, ...]
    start_frame: int
    end_frame: int


def load_tracking_table(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, names=RAW_COLUMNS, header=None, usecols=range(len(RAW_COLUMNS)))
    result = pd.DataFrame(
        {
            "frame": raw["frame"].astype(int),
            "track_id": raw["track_id"].astype(int),
            "class_id": raw["class_id"].astype(int),
            "cx": raw["left"].astype(float) + raw["width"].astype(float) / 2.0,
            "cy": raw["top"].astype(float) + raw["height"].astype(float) / 2.0,
        }
    )
    return result.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)


def _bindings_in_frame(frame_data: pd.DataFrame, role_types: Sequence[int]) -> list[tuple[int, ...]]:
    candidates_by_type: dict[int, list[int]] = {}
    for row in frame_data.itertuples(index=False):
        candidates_by_type.setdefault(int(row.class_id), []).append(int(row.track_id))
    role_candidates = [sorted(set(candidates_by_type.get(int(role_type), []))) for role_type in role_types]
    if any(not candidates for candidates in role_candidates):
        return []
    return sorted(
        tuple(int(track_id) for track_id in binding)
        for binding in product(*role_candidates)
        if len(set(binding)) == len(binding)
    )


def select_real_event(
    data: pd.DataFrame,
    role_types: Sequence[int],
    min_consecutive_frames: int,
) -> RealEvent:
    if min_consecutive_frames <= 0:
        raise ValueError("min_consecutive_frames must be positive")
    normalized_role_types = tuple(int(role_type) for role_type in role_types)
    active: dict[tuple[int, ...], tuple[int, int]] = {}

    for frame, frame_data in data.groupby("frame", sort=True):
        frame_id = int(frame)
        next_active: dict[tuple[int, ...], tuple[int, int]] = {}
        for binding in _bindings_in_frame(frame_data, normalized_role_types):
            previous = active.get(binding)
            start_frame = previous[0] if previous is not None and previous[1] == frame_id - 1 else frame_id
            next_active[binding] = (start_frame, frame_id)
            if frame_id - start_frame + 1 >= min_consecutive_frames:
                return RealEvent(normalized_role_types, binding, start_frame, frame_id)
        active = next_active

    raise ValueError(
        f"no binding for role types {normalized_role_types} persists for "
        f"{min_consecutive_frames} consecutive frames"
    )


def derive_query_from_event(
    data: pd.DataFrame,
    event: RealEvent,
    query_id: str,
    frame_width: int,
    frame_height: int,
    topk: int,
    theta_parts: int = 10,
    distance_parts: int = 8,
) -> QuerySpec:
    segment = data[
        data["frame"].between(event.start_frame, event.end_frame)
        & data["track_id"].isin(event.binding)
    ]
    positions_by_frame = {
        int(frame): {
            int(row.track_id): (float(row.cx), float(row.cy))
            for row in frame_data.itertuples(index=False)
        }
        for frame, frame_data in segment.groupby("frame", sort=True)
    }

    relations: dict[tuple[int, int], RelationConstraint] = {}
    for source_role in range(len(event.binding)):
        for target_role in range(source_role + 1, len(event.binding)):
            observed: list[tuple[int, int]] = []
            for frame in range(event.start_frame, event.end_frame + 1):
                positions = positions_by_frame.get(frame, {})
                source_id = event.binding[source_role]
                target_id = event.binding[target_role]
                if source_id not in positions or target_id not in positions:
                    raise ValueError("source event is not present in every frame of its interval")
                observed.append(
                    quantize_relation(
                        positions[source_id],
                        positions[target_id],
                        frame_width,
                        frame_height,
                        theta_parts,
                        distance_parts,
                    )
                )
            theta_bins = frozenset(theta for theta, _ in observed)
            distance_bins = [distance for _, distance in observed]
            relations[(source_role, target_role)] = RelationConstraint(
                min(distance_bins),
                max(distance_bins),
                theta_bins,
            )

    return QuerySpec(
        query_id=query_id,
        role_types=event.role_types,
        relations=relations,
        min_consecutive_frames=event.end_frame - event.start_frame + 1,
        topk=topk,
        theta_parts=theta_parts,
        distance_parts=distance_parts,
    )


def query_to_dict(dataset: str, event: RealEvent, query: QuerySpec) -> dict[str, object]:
    return {
        "dataset": dataset,
        "query_id": query.query_id,
        "role_types": list(query.role_types),
        "relations": [
            {
                "source_role": source_role,
                "target_role": target_role,
                "d_min": constraint.d_min,
                "d_max": constraint.d_max,
                "theta_bins": sorted(constraint.theta_bins),
            }
            for (source_role, target_role), constraint in sorted(query.relations.items())
        ],
        "min_consecutive_frames": query.min_consecutive_frames,
        "topk": query.topk,
        "theta_parts": query.theta_parts,
        "distance_parts": query.distance_parts,
        "source_event": asdict(event),
    }


def query_from_dict(payload: dict[str, object]) -> QuerySpec:
    relations = {
        (int(item["source_role"]), int(item["target_role"])): RelationConstraint(
            int(item["d_min"]),
            int(item["d_max"]),
            frozenset(int(theta) for theta in item["theta_bins"]),
        )
        for item in payload["relations"]
    }
    return QuerySpec(
        query_id=str(payload["query_id"]),
        role_types=tuple(int(role_type) for role_type in payload["role_types"]),
        relations=relations,
        min_consecutive_frames=int(payload["min_consecutive_frames"]),
        topk=int(payload["topk"]),
        theta_parts=int(payload["theta_parts"]),
        distance_parts=int(payload["distance_parts"]),
    )


def read_role_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "query_id": row["query_id"],
                    "role_types": tuple(int(value) for value in row["role_types"].split(";")),
                }
            )
    return rows


def generate_workload(
    roles_path: str | Path,
    output_path: str | Path,
    min_consecutive_frames: int = 10,
    topk: int = 10,
) -> list[dict[str, object]]:
    workload: list[dict[str, object]] = []
    data_cache: dict[str, pd.DataFrame] = {}
    for row in read_role_rows(roles_path):
        dataset = str(row["dataset"])
        if dataset not in data_cache:
            data_cache[dataset] = load_tracking_table(dataset_storage_path(dataset))
        data = data_cache[dataset]
        event = select_real_event(data, row["role_types"], min_consecutive_frames)
        width, height = dataset_frame_size(dataset)
        query = derive_query_from_event(
            data,
            event,
            query_id=str(row["query_id"]),
            frame_width=width,
            frame_height=height,
            topk=topk,
        )
        workload.append(query_to_dict(dataset, event, query))

    order = {name: index for index, name in enumerate(PREFERRED_VIDEO_ORDER)}
    workload.sort(key=lambda item: (order.get(str(item["dataset"]), 999), str(item["query_id"])))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(workload, indent=2, ensure_ascii=True), encoding="ascii")
    return workload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validated Q1/Q2 queries from real tracked events.")
    parser.add_argument(
        "--roles",
        type=Path,
        default=Path(__file__).with_name("corrected_baseline_roles.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "out" / "corrected_baseline_20260712" / "workload.json",
    )
    parser.add_argument("--df", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    workload = generate_workload(args.roles, args.output, args.df, args.topk)
    print(f"wrote {len(workload)} queries to {args.output}")


if __name__ == "__main__":
    main()
