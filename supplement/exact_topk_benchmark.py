from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from supplement.paper_experiment_utils import run_query_variant
from supplement.paper_query_benchmark import _load_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KS = (10, 20, 30, 40, 50)
WORKLOAD_DIRS = {
    "drtest": "paper_benchmark_figure7_topk_10_50_drtest",
    "drtrain": "paper_benchmark_figure7_topk_10_50_drtrain",
    "bdd100kA": "paper_benchmark_figure7_topk_10_50_bdd100kA",
    "bdd100kB": "paper_benchmark_figure7_topk_10_50_bdd100kB",
    "I-24": "paper_benchmark_figure7_topk_10_50_i24",
}


def _trimmed_mean(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) >= 3:
        ordered = ordered[1:-1]
    return sum(ordered) / len(ordered)


def _summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["video"]), int(row["k"]))].append(row)

    summaries = []
    for (video, k), group in sorted(grouped.items()):
        exhaustive_s = _trimmed_mean(row["exhaustive_s"] for row in group)
        optimized_s = _trimmed_mean(row["optimized_s"] for row in group)
        summaries.append(
            {
                "video": video,
                "k": k,
                "query_count": len(group),
                "exhaustive_s": exhaustive_s,
                "optimized_s": optimized_s,
                "speedup": exhaustive_s / optimized_s if optimized_s else 0.0,
                "processed_window_ratio": _trimmed_mean(
                    row["processed_window_ratio"] for row in group
                ),
                "bound_setup_s": _trimmed_mean(row["bound_setup_s"] for row in group),
                "early_termination_rate": sum(bool(row["early_terminated"]) for row in group)
                / len(group),
                "all_exact": all(bool(row["exact_match"]) for row in group),
            }
        )
    return summaries


def _read_workload(video: str, limit_queries: int) -> list[dict]:
    source = (
        REPO_ROOT
        / "supplement"
        / "out"
        / WORKLOAD_DIRS[video]
        / "paper_query_time_raw.csv"
    )
    queries: list[dict] = []
    with source.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["topk"]) != 10:
                continue
            queries.append(
                {
                    "query_types": [int(value) for value in row["query_types"].split(",")],
                    "theta_lt": float(row["theta_lt"]),
                    "d_lt": float(row["d_lt"]),
                    "frame_threshold": int(row["frame_threshold"]),
                }
            )
            if len(queries) >= limit_queries:
                break
    if len(queries) < limit_queries:
        raise ValueError(f"only {len(queries)} workload queries available for {video}")
    return queries


def _canonical_top_paths(run) -> list[tuple[tuple[int, ...], tuple[int, ...], int]]:
    return [
        (tuple(binding), tuple(sorted(frames)), int(score))
        for binding, frames, score in run.top_paths
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser("Benchmark exact Top-k upper-bound pruning")
    parser.add_argument("--videos", default=",".join(WORKLOAD_DIRS))
    parser.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    parser.add_argument("--limit_queries", type=int, default=5)
    parser.add_argument(
        "--out_dir",
        default="supplement/out/exact_topk_benchmark",
    )
    args = parser.parse_args()

    videos = [value.strip() for value in args.videos.split(",") if value.strip()]
    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]
    out_dir = REPO_ROOT / args.out_dir
    raw_path = out_dir / "exact_topk_raw.csv"
    summary_path = out_dir / "exact_topk_summary.csv"
    raw_rows: list[dict] = []

    for video in videos:
        print(f"Loading {video}", flush=True)
        index, windowid_index, cp_graphs = _load_artifacts(video, 0.7)
        queries = _read_workload(video, args.limit_queries)
        for query_id, query in enumerate(queries, start=1):
            for k_index, k in enumerate(ks):
                kwargs = dict(
                    index=index,
                    cp_graphs=cp_graphs,
                    windowid_index=windowid_index,
                    query_object_types=query["query_types"],
                    query_spatial=(query["theta_lt"], query["d_lt"]),
                    topk=k,
                    frame_threshold=query["frame_threshold"],
                    variant="full",
                )
                if (query_id + k_index) % 2:
                    optimized = run_query_variant(**kwargs, exact_topk_pruning=True)
                    exhaustive = run_query_variant(**kwargs)
                else:
                    exhaustive = run_query_variant(**kwargs)
                    optimized = run_query_variant(**kwargs, exact_topk_pruning=True)

                exhaustive_paths = _canonical_top_paths(exhaustive)
                optimized_paths = _canonical_top_paths(optimized)
                exact_match = optimized_paths == exhaustive_paths
                if not exact_match:
                    raise AssertionError(
                        f"Top-k mismatch video={video} query={query_id} k={k}"
                    )
                if len(exhaustive_paths) < k:
                    raise AssertionError(
                        f"insufficient results video={video} query={query_id} k={k}"
                    )

                candidate_windows = exhaustive.stats.windows_after_type_filter
                row = {
                    "video": video,
                    "query_id": query_id,
                    "k": k,
                    "query_types": ",".join(map(str, query["query_types"])),
                    "theta_lt": query["theta_lt"],
                    "d_lt": query["d_lt"],
                    "frame_threshold": query["frame_threshold"],
                    "exhaustive_s": exhaustive.stage_times.total_s,
                    "optimized_s": optimized.stage_times.total_s,
                    "speedup": exhaustive.stage_times.total_s / optimized.stage_times.total_s,
                    "candidate_windows": candidate_windows,
                    "processed_windows": optimized.candidate_windows_processed,
                    "processed_window_ratio": optimized.candidate_windows_processed
                    / candidate_windows,
                    "bound_setup_s": optimized.bound_setup_s,
                    "early_terminated": optimized.early_terminated,
                    "exact_match": exact_match,
                }
                raw_rows.append(row)
                _write_csv(raw_path, raw_rows)
                print(
                    f"{video} q={query_id} k={k} "
                    f"full={row['exhaustive_s']:.4f}s opt={row['optimized_s']:.4f}s "
                    f"windows={row['processed_windows']}/{candidate_windows} "
                    f"speedup={row['speedup']:.2f}x",
                    flush=True,
                )

    summaries = _summarize_rows(raw_rows)
    _write_csv(summary_path, summaries)
    print(f"Wrote {raw_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
