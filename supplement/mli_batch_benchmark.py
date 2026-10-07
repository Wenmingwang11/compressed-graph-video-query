from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_query_pairs import run_single_query_with_pair_constraints


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_query_specs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _stats_value(stats: Any, name: str) -> int:
    return int(getattr(stats, name, 0))


def main() -> None:
    parser = argparse.ArgumentParser("Run batch benchmarks for the MLI structured-query method")
    parser.add_argument(
        "--query_specs",
        default=str(REPO_ROOT / "supplement" / "baseline_formal_queries.csv"),
        help="CSV file with query specs",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "supplement" / "out" / "mli_batch_benchmark.csv"),
        help="output CSV path",
    )
    parser.add_argument(
        "--query_ids",
        default="",
        help="optional comma-separated query_id filter",
    )
    parser.add_argument(
        "--s",
        type=float,
        default=0.7,
        help="window similarity threshold used by the prebuilt index",
    )
    parser.add_argument(
        "--max_bindings_per_window",
        type=int,
        default=0,
        help="0 means no cap",
    )
    args = parser.parse_args()

    query_specs = _read_query_specs(Path(args.query_specs))
    query_id_filter = {token.strip() for token in args.query_ids.split(",") if token.strip()}
    if query_id_filter:
        query_specs = [spec for spec in query_specs if spec["query_id"] in query_id_filter]

    rows: list[dict[str, Any]] = []
    total_runs = len(query_specs)
    for idx, spec in enumerate(query_specs, start=1):
        query_types = [int(token.strip()) for token in spec["types"].split(",") if token.strip()]
        print(
            f"[{idx}/{total_runs}] method=mli query_id={spec['query_id']} dataset={spec['dataset']}",
            flush=True,
        )
        elapsed, stats, top_paths, pair_constraints = run_single_query_with_pair_constraints(
            video_name=spec["dataset"],
            s=args.s,
            query_types=query_types,
            pair_constraints_text=spec["pair_constraints"],
            topk=int(spec["topk"]),
            frame_threshold=int(spec["frame_threshold"]),
            max_bindings_per_window=args.max_bindings_per_window,
        )

        top1_path: list[int] | None = None
        top1_frames: list[int] = []
        top1_frame_count = 0
        if top_paths:
            top1_path, top1_frames, top1_frame_count = top_paths[0]

        row = {
            "method": "mli",
            "query_id": spec["query_id"],
            "dataset": spec["dataset"],
            "types": spec["types"],
            "pair_constraints": spec["pair_constraints"],
            "parsed_pair_constraints": str(pair_constraints),
            "frame_threshold": int(spec["frame_threshold"]),
            "topk": int(spec["topk"]),
            "temporal_mode": spec["temporal_mode"],
            "elapsed_s": float(elapsed),
            "windows_total": _stats_value(stats, "windows_total"),
            "windows_after_type_filter": _stats_value(stats, "windows_after_type_filter"),
            "candidate_vertices_total": _stats_value(stats, "candidate_vertices_total"),
            "bindings_total": _stats_value(stats, "bindings_total"),
            "bindings_matched": _stats_value(stats, "bindings_matched"),
            "pair_lookups": _stats_value(stats, "pair_lookups"),
            "theta_d_bins_scanned": _stats_value(stats, "theta_d_bins_scanned"),
            "match_count": len(top_paths),
            "top1_path": "" if top1_path is None else ",".join(map(str, top1_path)),
            "top1_frame_count": int(top1_frame_count),
            "top1_frames": "" if not top1_frames else ",".join(map(str, top1_frames)),
            "note": spec.get("note", ""),
        }
        rows.append(row)
        print(
            f"  elapsed={row['elapsed_s']:.4f}s windows={row['windows_after_type_filter']}/{row['windows_total']} "
            f"bindings={row['bindings_matched']}/{row['bindings_total']}",
            flush=True,
        )

    out_path = Path(args.out)
    _ensure_parent_dir(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved_csv={out_path}", flush=True)


if __name__ == "__main__":
    main()
