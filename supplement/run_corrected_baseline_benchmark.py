from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from supplement.corrected_baseline_strategies import build_strategy
from supplement.corrected_query_model import QueryResult, QuerySpec, exact_query, result_signature
from supplement.dataset_config import PREFERRED_VIDEO_ORDER, dataset_frame_size, dataset_storage_path
from supplement.generate_corrected_baseline_workload import load_tracking_table, query_from_dict
from paper_query_corrected import build_corrected_mli, corrected_mli_tag
from vsimsearch.corrected_mli_querying import query_corrected_mli
from vsimsearch.mul_build_index import artifact_paths


METHOD_TO_STRATEGY = {
    "VQPy-style": "vqpy",
    "EVA-style": "eva",
    "VOCAL-UDF-style": "vocal-udf",
    "Video-ColBERT-style": "video-colbert",
    "Seiden-style": "seiden",
    "LAVA-style": "lava",
    "STAR-style": "star",
}
METHODS = ("Ours", *METHOD_TO_STRATEGY)
RAW_FIELDS = (
    "dataset",
    "query_id",
    "method",
    "repetition",
    "elapsed_s",
    "signature",
    "signature_verified",
)
SUMMARY_FIELDS = (
    "dataset",
    "query_id",
    "method",
    "median_s",
    "q1_s",
    "q3_s",
    "source",
    "signature",
    "signature_verified",
)


class _OursStrategy:
    def __init__(
        self,
        dataset: str,
        data,
        frame_width: int,
        frame_height: int,
        similarity: float = 0.7,
    ) -> None:
        self._data = data
        self._frame_width = int(frame_width)
        self._frame_height = int(frame_height)
        tag = corrected_mli_tag()
        index_path, cp_graphs_path, window_edges_path, _ = artifact_paths(
            dataset, similarity, output_tag=tag
        )
        if not index_path.exists() or not cp_graphs_path.exists() or not window_edges_path.exists():
            build_corrected_mli(dataset, similarity)
        with index_path.open("rb") as handle:
            self._index = pickle.load(handle)
        with cp_graphs_path.open("rb") as handle:
            self._cp_graphs = pickle.load(handle)
        with window_edges_path.open("rb") as handle:
            self._window_edges = pickle.load(handle)

    def query(self, query: QuerySpec, *, stats: dict | None = None,
              return_all: bool = False) -> list[QueryResult]:
        return query_corrected_mli(
            self._index,
            self._cp_graphs,
            self._data,
            query,
            self._frame_width,
            self._frame_height,
            legacy_single_direction=True,
            indexed_frame_width=self._frame_width,
            indexed_frame_height=self._frame_height,
            window_edge_indexes=self._window_edges,
            stats=stats,
            return_all=return_all,
        )

    def release(self) -> None:
        self._index = None
        self._cp_graphs = None
        self._window_edges = None
        self._data = None


def measure_query(
    strategy,
    query,
    expected_signature: str,
    repetitions: int = 5,
    warmups: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict[str, object]]:
    if repetitions <= 0 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups must be non-negative")

    for _ in range(warmups):
        warmup_results = strategy.query(query)
        if result_signature(warmup_results) != expected_signature:
            raise RuntimeError("result signature mismatch during warm-up")

    rows: list[dict[str, object]] = []
    for repetition in range(1, repetitions + 1):
        start = clock()
        results = strategy.query(query)
        elapsed = clock() - start
        signature = result_signature(results)
        if signature != expected_signature:
            raise RuntimeError(
                f"result signature mismatch: expected={expected_signature}, actual={signature}"
            )
        rows.append(
            {
                "repetition": repetition,
                "elapsed_s": float(elapsed),
                "signature": signature,
                "signature_verified": True,
            }
        )
    return rows


def summarize_measurements(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["query_id"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    dataset_order = {dataset: index for index, dataset in enumerate(PREFERRED_VIDEO_ORDER)}
    method_order = {method: index for index, method in enumerate(METHODS)}
    for (dataset, query_id, method), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            dataset_order.get(item[0][0], 999),
            method_order.get(item[0][2], 999),
        ),
    ):
        values = np.asarray([float(row["elapsed_s"]) for row in group], dtype=float)
        signatures = {str(row["signature"]) for row in group}
        verified = all(bool(row["signature_verified"]) for row in group)
        if len(signatures) != 1 or not verified:
            raise RuntimeError(f"unverified or inconsistent measurements for {(dataset, query_id, method)}")
        summary.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "method": method,
                "median_s": float(np.median(values)),
                "q1_s": float(np.quantile(values, 0.25)),
                "q3_s": float(np.quantile(values, 0.75)),
                "source": "measured",
                "signature": next(iter(signatures)),
                "signature_verified": True,
            }
        )
    return summary


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_workload(path: Path) -> dict[str, list[tuple[str, QuerySpec]]]:
    payload = json.loads(path.read_text(encoding="ascii"))
    grouped: dict[str, list[tuple[str, QuerySpec]]] = {}
    for item in payload:
        grouped.setdefault(str(item["dataset"]), []).append((str(item["query_id"]), query_from_dict(item)))
    for queries in grouped.values():
        queries.sort(key=lambda item: item[0])
    return grouped


def run_benchmark(
    workload_path: str | Path,
    output_dir: str | Path,
    datasets: Iterable[str] = PREFERRED_VIDEO_ORDER,
    methods: Iterable[str] = METHODS,
    repetitions: int = 5,
    warmups: int = 1,
    resume: bool = False,
) -> tuple[Path, Path]:
    workload_file = Path(workload_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    workload = _load_workload(workload_file)
    selected_datasets = tuple(datasets)
    selected_methods = tuple(methods)
    unknown = sorted(set(selected_methods).difference(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")

    raw_rows: list[dict[str, object]] = []
    oracle_manifest: dict[str, dict[str, str]] = {}
    raw_path = destination / "raw_measurements.csv"
    summary_path = destination / "summary.csv"
    manifest_path = destination / "run_manifest.json"
    for stale_path in (summary_path, manifest_path):
        stale_path.unlink(missing_ok=True)
    if resume and raw_path.exists():
        with raw_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_rows.append(
                    {
                        **row,
                        "repetition": int(row["repetition"]),
                        "elapsed_s": float(row["elapsed_s"]),
                        "signature_verified": row["signature_verified"].strip().lower()
                        in {"1", "true", "yes"},
                    }
                )
    else:
        raw_path.unlink(missing_ok=True)

    for dataset_index, dataset in enumerate(selected_datasets):
        if dataset not in workload:
            raise ValueError(f"workload has no queries for dataset {dataset}")
        data_path = dataset_storage_path(dataset)
        data = load_tracking_table(data_path)
        width, height = dataset_frame_size(dataset)
        queries = workload[dataset]

        expected: dict[str, str] = {}
        oracle_manifest[dataset] = {}
        for query_id, query in queries:
            signature = result_signature(exact_query(data, query, width, height))
            expected[query_id] = signature
            oracle_manifest[dataset][query_id] = signature

        def group_complete(query_id: str, method: str) -> bool:
            group = [
                row
                for row in raw_rows
                if row["dataset"] == dataset
                and row["query_id"] == query_id
                and row["method"] == method
            ]
            return (
                len(group) == repetitions
                and {int(row["repetition"]) for row in group} == set(range(1, repetitions + 1))
                and {str(row["signature"]) for row in group} == {expected[query_id]}
                and all(bool(row["signature_verified"]) for row in group)
            )

        rotation = dataset_index % len(selected_methods)
        method_order = selected_methods[rotation:] + selected_methods[:rotation]
        for method_index, method in enumerate(method_order):
            if all(group_complete(query_id, method) for query_id, _ in queries):
                print(f"resume: keeping {dataset} all queries {method}", flush=True)
                continue
            if method == "Ours":
                strategy = _OursStrategy(dataset, data, width, height)
            else:
                allowed_type_pairs = {
                    (query.role_types[source_role], query.role_types[target_role])
                    for _, query in queries
                    for source_role, target_role in query.relations
                }
                strategy = build_strategy(
                    METHOD_TO_STRATEGY[method],
                    data,
                    width,
                    height,
                    allowed_type_pairs=allowed_type_pairs if method == "STAR-style" else None,
                )

            query_order = queries if method_index % 2 == 0 else list(reversed(queries))
            for query_id, query in query_order:
                group_rows = [
                    row
                    for row in raw_rows
                    if row["dataset"] == dataset
                    and row["query_id"] == query_id
                    and row["method"] == method
                ]
                if group_complete(query_id, method):
                    print(f"resume: keeping {dataset} {query_id} {method}", flush=True)
                    continue
                if group_rows:
                    raw_rows = [row for row in raw_rows if row not in group_rows]
                measured = measure_query(
                    strategy,
                    query,
                    expected[query_id],
                    repetitions=repetitions,
                    warmups=warmups,
                )
                for row in measured:
                    raw_rows.append(
                        {
                            "dataset": dataset,
                            "query_id": query_id,
                            "method": method,
                            **row,
                        }
                    )
                _write_csv(raw_path, raw_rows, RAW_FIELDS)
                print(f"measured: {dataset} {query_id} {method}", flush=True)

            release = getattr(strategy, "release", None)
            if release is not None:
                release()
            del strategy
            gc.collect()

        del data
        gc.collect()

    summary_rows = summarize_measurements(raw_rows)
    expected_groups = {
        (dataset, query_id, method)
        for dataset in selected_datasets
        for query_id, _ in workload[dataset]
        for method in selected_methods
    }
    actual_groups = {
        (str(row["dataset"]), str(row["query_id"]), str(row["method"])) for row in raw_rows
    }
    if actual_groups != expected_groups:
        raise RuntimeError(
            f"benchmark group coverage mismatch: missing={sorted(expected_groups - actual_groups)}, "
            f"extra={sorted(actual_groups - expected_groups)}"
        )
    _write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    (destination / "unsupported_methods.json").write_text(
        json.dumps(
            {
                "RELOCATE": "native query unit is single-object image/track localization, not strict multi-object QST",
                "SketchQL": "native query unit is trajectory sketch similarity, not strict category-relation QST",
            },
            indent=2,
        ),
        encoding="ascii",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "workload": str(workload_file.resolve()),
                "workload_sha256": _sha256(workload_file),
                "datasets": list(selected_datasets),
                "dataset_sha256": {
                    dataset: _sha256(dataset_storage_path(dataset)) for dataset in selected_datasets
                },
                "methods": list(selected_methods),
                "warmups": warmups,
                "repetitions": repetitions,
                "timing_scope": "preloaded online query only; artifact construction and data loading excluded",
                "oracle_signatures": oracle_manifest,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            indent=2,
        ),
        encoding="ascii",
    )
    return raw_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the corrected Q1/Q2 adapted-baseline benchmark.")
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path(__file__).parent / "out" / "corrected_baseline_20260712" / "workload.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "out" / "corrected_baseline_20260712",
    )
    parser.add_argument("--datasets", default=",".join(PREFERRED_VIDEO_ORDER))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raw_path, summary_path = run_benchmark(
        args.workload,
        args.output_dir,
        datasets=[value.strip() for value in args.datasets.split(",") if value.strip()],
        methods=[value.strip() for value in args.methods.split(",") if value.strip()],
        repetitions=args.repetitions,
        warmups=args.warmups,
        resume=args.resume,
    )
    print(raw_path)
    print(summary_path)


if __name__ == "__main__":
    main()
