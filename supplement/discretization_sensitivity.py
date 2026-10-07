from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supplement.paper_experiment_utils import (
    ensure_parent_dir,
    run_query_variant,
    sample_query_types,
    sample_thresholds_from_cp_graphs,
)
from vsimsearch.mul_build_index import artifact_paths, discretization_tag, main as build_index


@dataclass
class DiscretizationRow:
    video: str
    s: str
    theta_parts: int
    d_parts: int
    query_id: int
    query_types: str
    theta_lt: float
    d_lt: float
    frame_threshold: int
    topk: int
    index_file_bytes: int
    cp_graphs_file_bytes: int
    windowid_index_file_bytes: int
    index_total_bytes: int
    windows_total: int
    windows_after_type_filter: int
    candidate_vertices_total: int
    bindings_total: int
    bindings_matched: int
    t_type_filter_s: float
    t_candidate_vertex_s: float
    t_spatial_match_s: float
    t_prefix_tree_s: float
    t_total_s: float
    top_path_count: int
    top1_frame_count: int


def parse_granularities(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"invalid granularity token: {token}")
        theta_raw, d_raw = token.split(":", 1)
        pairs.append((int(theta_raw), int(d_raw)))
    return pairs


def _write_csv(path: Path, rows: list[dict]):
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_csv_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    ensure_parent_dir(path)
    fieldnames = list(rows[0].keys())
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_stat_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _sample_allowed_query_types(
    index: dict,
    obj_num: int,
    rng: random.Random,
    allowed_types: Optional[set[int]],
) -> list[int]:
    if not allowed_types:
        return sample_query_types(index, obj_num, rng)

    filtered_index = {
        window_id: {
            node_type: node_ids
            for node_type, node_ids in window_data.items()
            if node_type in allowed_types
        }
        for window_id, window_data in index.items()
    }
    filtered_index = {window_id: window_data for window_id, window_data in filtered_index.items() if window_data}
    return sample_query_types(filtered_index, obj_num, rng)


def _ensure_artifacts(video: str, s: float, theta_parts: int, d_parts: int):
    tag = discretization_tag(theta_parts, d_parts)
    index_path, cp_graphs_path, windowid_index_path, _ = artifact_paths(video, s, output_tag=tag)
    if index_path.exists() and cp_graphs_path.exists() and windowid_index_path.exists():
        return

    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(REPO_ROOT / "vsimsearch")
        build_index(video, s, theta_n_parts=theta_parts, theta_d_parts=d_parts, output_tag=tag)
    finally:
        os.chdir(old_cwd)


def _load_artifacts(video: str, s: float, theta_parts: int, d_parts: int):
    tag = discretization_tag(theta_parts, d_parts)
    index_path, cp_graphs_path, windowid_index_path, _ = artifact_paths(video, s, output_tag=tag)
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    with open(windowid_index_path, "rb") as f:
        windowid_index = pickle.load(f)
    with open(cp_graphs_path, "rb") as f:
        cp_graphs = pickle.load(f)
    return index, windowid_index, cp_graphs, index_path, cp_graphs_path, windowid_index_path


def _summarize_rows(rows: list[DiscretizationRow]) -> list[dict]:
    groups: dict[tuple[str, int, int], list[DiscretizationRow]] = {}
    for row in rows:
        groups.setdefault((row.video, row.theta_parts, row.d_parts), []).append(row)

    summary: list[dict] = []
    for (video, theta_parts, d_parts), group in groups.items():
        n = len(group)
        summary.append(
            {
                "video": video,
                "theta_parts": theta_parts,
                "d_parts": d_parts,
                "query_count": n,
                "index_file_bytes": group[0].index_file_bytes,
                "cp_graphs_file_bytes": group[0].cp_graphs_file_bytes,
                "windowid_index_file_bytes": group[0].windowid_index_file_bytes,
                "index_total_bytes": group[0].index_total_bytes,
                "avg_windows_after_type_filter": sum(r.windows_after_type_filter for r in group) / n,
                "avg_candidate_vertices_total": sum(r.candidate_vertices_total for r in group) / n,
                "avg_bindings_total": sum(r.bindings_total for r in group) / n,
                "avg_bindings_matched": sum(r.bindings_matched for r in group) / n,
                "avg_t_type_filter_s": sum(r.t_type_filter_s for r in group) / n,
                "avg_t_candidate_vertex_s": sum(r.t_candidate_vertex_s for r in group) / n,
                "avg_t_spatial_match_s": sum(r.t_spatial_match_s for r in group) / n,
                "avg_t_prefix_tree_s": sum(r.t_prefix_tree_s for r in group) / n,
                "avg_t_total_s": sum(r.t_total_s for r in group) / n,
                "avg_top1_frame_count": sum(r.top1_frame_count for r in group) / n,
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser("Discretization sensitivity benchmark")
    parser.add_argument("--videos", default="drtest,drtrain")
    parser.add_argument("--s", type=float, default=0.7)
    parser.add_argument("--granularities", default="6:4,8:6,10:8,12:10")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--obj_num", type=int, default=3)
    parser.add_argument("--frame_threshold", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed_types", default="0,1,2,3,5,7,8")
    parser.add_argument("--build_missing", action="store_true")
    parser.add_argument("--out_dir", default="supplement/out/discretization_sensitivity")
    args = parser.parse_args()

    videos = [token.strip() for token in args.videos.split(",") if token.strip()]
    granularities = parse_granularities(args.granularities)
    allowed_types = {
        int(token.strip()) for token in args.allowed_types.split(",") if token.strip()
    } or None
    rng = random.Random(args.seed)

    out_dir = REPO_ROOT / args.out_dir
    raw_path = out_dir / "discretization_sensitivity_raw.csv"
    summary_path = out_dir / "discretization_sensitivity_summary.csv"
    progress_path = out_dir / "discretization_sensitivity_progress.log"

    for path in (raw_path, summary_path, progress_path):
        if path.exists():
            path.unlink()

    rows: list[DiscretizationRow] = []
    for video in videos:
        for theta_parts, d_parts in granularities:
            if args.build_missing:
                _ensure_artifacts(video, args.s, theta_parts, d_parts)

            (
                index,
                windowid_index,
                cp_graphs,
                index_path,
                cp_graphs_path,
                windowid_index_path,
            ) = _load_artifacts(video, args.s, theta_parts, d_parts)
            index_bytes = _safe_stat_bytes(index_path)
            cp_bytes = _safe_stat_bytes(cp_graphs_path)
            windowid_bytes = _safe_stat_bytes(windowid_index_path)
            total_bytes = index_bytes + cp_bytes + windowid_bytes

            for query_id in range(1, args.n + 1):
                query_types = _sample_allowed_query_types(index, args.obj_num, rng, allowed_types)
                theta_lt, d_lt = sample_thresholds_from_cp_graphs(cp_graphs, rng)
                run = run_query_variant(
                    index=index,
                    cp_graphs=cp_graphs,
                    windowid_index=windowid_index,
                    query_object_types=query_types,
                    query_spatial=(theta_lt, d_lt),
                    topk=args.topk,
                    frame_threshold=args.frame_threshold,
                    variant="full",
                )
                top1_frame_count = run.top_paths[0][2] if run.top_paths else 0
                rows.append(
                    DiscretizationRow(
                        video=video,
                        s=str(args.s),
                        theta_parts=theta_parts,
                        d_parts=d_parts,
                        query_id=query_id,
                        query_types=",".join(map(str, query_types)),
                        theta_lt=float(theta_lt),
                        d_lt=float(d_lt),
                        frame_threshold=args.frame_threshold,
                        topk=args.topk,
                        index_file_bytes=index_bytes,
                        cp_graphs_file_bytes=cp_bytes,
                        windowid_index_file_bytes=windowid_bytes,
                        index_total_bytes=total_bytes,
                        windows_total=run.stats.windows_total,
                        windows_after_type_filter=run.stats.windows_after_type_filter,
                        candidate_vertices_total=run.stats.candidate_vertices_total,
                        bindings_total=run.stats.bindings_total,
                        bindings_matched=run.stats.bindings_matched,
                        t_type_filter_s=run.stage_times.type_filter_s,
                        t_candidate_vertex_s=run.stage_times.candidate_vertex_s,
                        t_spatial_match_s=run.stage_times.spatial_match_s,
                        t_prefix_tree_s=run.stage_times.prefix_tree_s,
                        t_total_s=run.stage_times.total_s,
                        top_path_count=len(run.top_paths),
                        top1_frame_count=top1_frame_count,
                    )
                )
                row_dict = asdict(rows[-1])
                _append_csv_rows(raw_path, [row_dict])
                line = (
                    f"{video} theta_parts={theta_parts} d_parts={d_parts} "
                    f"query_id={query_id}/{args.n} t_total_s={run.stage_times.total_s:.4f}"
                )
                with progress_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                print(line, flush=True)

    _write_csv(summary_path, _summarize_rows(rows))


if __name__ == "__main__":
    main()
