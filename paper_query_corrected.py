from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from supplement.dataset_config import dataset_frame_size
from supplement.dataset_config import dataset_storage_path
from supplement.generate_corrected_baseline_workload import load_tracking_table
from vsimsearch.mul_build_index import artifact_paths, discretization_tag, main as build_mli


def corrected_mli_tag(theta_parts: int = 10, distance_parts: int = 8) -> str:
    return f"corrected_mli_dims_{discretization_tag(theta_parts, distance_parts)}"


def build_all_observed_window_type_index(
    data: pd.DataFrame,
    key_frame_indices: list[int],
) -> dict[int, dict[int, set[int]]]:
    if not key_frame_indices:
        return {}
    selected = data[["frame", "track_id", "class_id"]].drop_duplicates()
    starts = np.asarray(key_frame_indices, dtype=np.int64)
    window_offsets = np.searchsorted(
        starts,
        selected["frame"].to_numpy(dtype=np.int64),
        side="right",
    ) - 1
    selected = selected.loc[window_offsets >= 0].copy()
    selected["window_id"] = window_offsets[window_offsets >= 0] + 1

    index: dict[int, dict[int, set[int]]] = {}
    for window_id, class_id, track_id in selected[
        ["window_id", "class_id", "track_id"]
    ].itertuples(index=False, name=None):
        index.setdefault(int(window_id), {}).setdefault(int(class_id), set()).add(
            int(track_id)
        )
    return index


def repair_corrected_type_index(
    dataset: str,
    similarity: float = 0.7,
    theta_parts: int = 10,
    distance_parts: int = 8,
) -> Path:
    keyframes_path = (
        Path(__file__).resolve().parent
        / "storage"
        / "zhunbei"
        / f"{dataset}_{similarity}_subsection.pkl"
    )
    with keyframes_path.open("rb") as handle:
        keyframes = pickle.load(handle)
    data = load_tracking_table(dataset_storage_path(dataset))
    index = build_all_observed_window_type_index(data, list(keyframes))
    index_path, _, _, _ = artifact_paths(
        dataset,
        similarity,
        output_tag=corrected_mli_tag(theta_parts, distance_parts),
    )
    with index_path.open("wb") as handle:
        pickle.dump(index, handle)
    return index_path


def build_corrected_mli(
    dataset: str,
    similarity: float = 0.7,
    theta_parts: int = 10,
    distance_parts: int = 8,
) -> tuple[Path, Path, Path, Path]:
    width, height = dataset_frame_size(dataset)
    paths = build_mli(
        dataset,
        similarity,
        theta_n_parts=theta_parts,
        theta_d_parts=distance_parts,
        output_tag=corrected_mli_tag(theta_parts, distance_parts),
        frame_width=width,
        frame_height=height,
        edge_build_mode="normal",
    )
    repair_corrected_type_index(
        dataset,
        similarity,
        theta_parts,
        distance_parts,
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the corrected directed MLI artifacts.")
    parser.add_argument("dataset")
    parser.add_argument("--similarity", type=float, default=0.7)
    parser.add_argument("--theta-parts", type=int, default=10)
    parser.add_argument("--distance-parts", type=int, default=8)
    parser.add_argument("--repair-types-only", action="store_true")
    args = parser.parse_args()
    if args.repair_types_only:
        print(
            repair_corrected_type_index(
                args.dataset,
                args.similarity,
                args.theta_parts,
                args.distance_parts,
            )
        )
    else:
        for path in build_corrected_mli(
            args.dataset,
            args.similarity,
            args.theta_parts,
            args.distance_parts,
        ):
            print(path)


if __name__ == "__main__":
    main()
