from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supplement.paper_experiment_utils import (
    compute_set_metrics,
    ensure_parent_dir,
    result_frame_units,
    run_query_variant,
    sample_query_types,
    sample_thresholds_from_cp_graphs,
)
from vsimsearch.index_files import resolve_index_paths


@dataclass
class AblationRow:
    video: str
    s: str
    obj_num: int
    query_types: str
    theta_lt: float
    d_lt: float
    variant: str
    frame_threshold: int
    topk: int
    windows_total: int
    windows_after_type_filter: int
    candidate_vertices_total: int
    bindings_total: int
    bindings_matched: int
    pair_lookups: int
    theta_d_bins_scanned: int
    t_type_filter_s: float
    t_candidate_vertex_s: float
    t_spatial_match_s: float
    t_prefix_tree_s: float
    t_total_s: float
    result_unit_count: int
    top_path_count: int
    top1_frame_count: int
    precision: float
    recall: float
    f1: float
    jaccard: float


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    try:
        import os

        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def _write_csv(path: Path, rows: list[dict]):
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _filter_index_by_allowed_types(index: dict, allowed_types: Optional[set[int]]) -> dict:
    if not allowed_types:
        return index
    filtered = {
        window_id: {
            node_type: node_ids
            for node_type, node_ids in window_data.items()
            if node_type in allowed_types
        }
        for window_id, window_data in index.items()
    }
    return {window_id: window_data for window_id, window_data in filtered.items() if window_data}


def _parse_allowed_types_by_obj_num(raw: str) -> dict[int, set[int]]:
    mapping: dict[int, set[int]] = {}
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"invalid allowed_types_by_obj_num token: {token}")
        obj_num_raw, types_raw = token.split(":", 1)
        obj_num = int(obj_num_raw.strip())
        mapping[obj_num] = {
            int(type_token.strip())
            for type_token in types_raw.split(",")
            if type_token.strip()
        }
    return mapping


def _load_artifacts_for_s(video: str, s: Optional[float]):
    index_path, windowid_index_path, cp_graphs_path = resolve_index_paths(video, s)
    if cp_graphs_path is None:
        raise FileNotFoundError(f"missing compressed graph index for video={video}, s={s}")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    with open(windowid_index_path, "rb") as f:
        windowid_index = pickle.load(f)
    with open(cp_graphs_path, "rb") as f:
        cp_graphs = pickle.load(f)
    return index, windowid_index, cp_graphs


def _tag_paths(video: str, tag: str) -> tuple[Path, Path, Path]:
    base = REPO_ROOT / "storage" / "index"
    return (
        base / f"{video}_{tag}_index.pkl",
        base / f"{video}_{tag}_windowid_index.pkl",
        base / f"{video}_{tag}_cp_graphs.pkl",
    )


def _load_artifacts_for_tag(video: str, tag: str):
    index_path, windowid_index_path, cp_graphs_path = _tag_paths(video, tag)
    if not index_path.exists() or not windowid_index_path.exists() or not cp_graphs_path.exists():
        raise FileNotFoundError(f"missing tag artifacts for video={video}, tag={tag}")
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    with open(windowid_index_path, "rb") as f:
        windowid_index = pickle.load(f)
    with open(cp_graphs_path, "rb") as f:
        cp_graphs = pickle.load(f)
    return index, windowid_index, cp_graphs


def _ensure_no_split_index(video: str, tag: str = "nosplit"):
    index_path, windowid_index_path, cp_graphs_path = _tag_paths(video, tag)
    if index_path.exists() and windowid_index_path.exists() and cp_graphs_path.exists():
        return

    subsection_path = REPO_ROOT / "storage" / "zhunbei" / f"{video}_{tag}_subsection.pkl"
    subsection_path.parent.mkdir(parents=True, exist_ok=True)
    if not subsection_path.exists():
        with subsection_path.open("wb") as f:
            pickle.dump([1], f)

    (REPO_ROOT / "storage" / "index" / "build_cp_graph_time").mkdir(parents=True, exist_ok=True)
    with pushd(REPO_ROOT / "vsimsearch"):
        from vsimsearch.mul_build_index import main as build_index

        build_index(video, tag)


def _summarize_rows(rows: list[AblationRow]) -> list[dict]:
    groups: dict[tuple[str, str], list[AblationRow]] = {}
    for row in rows:
        groups.setdefault((row.video, row.variant), []).append(row)

    summary_rows: list[dict] = []
    for (video, variant), group in groups.items():
        n = len(group)
        summary_rows.append(
            {
                "video": video,
                "variant": variant,
                "query_count": n,
                "avg_windows_after_type_filter": sum(r.windows_after_type_filter for r in group) / n,
                "avg_candidate_vertices_total": sum(r.candidate_vertices_total for r in group) / n,
                "avg_bindings_total": sum(r.bindings_total for r in group) / n,
                "avg_bindings_matched": sum(r.bindings_matched for r in group) / n,
                "avg_t_total_s": sum(r.t_total_s for r in group) / n,
                "avg_result_unit_count": sum(r.result_unit_count for r in group) / n,
                "avg_top1_frame_count": sum(r.top1_frame_count for r in group) / n,
                "avg_precision": sum(r.precision for r in group) / n,
                "avg_recall": sum(r.recall for r in group) / n,
                "avg_f1": sum(r.f1 for r in group) / n,
                "avg_jaccard": sum(r.jaccard for r in group) / n,
            }
        )
    return summary_rows


def main():
    parser = argparse.ArgumentParser("Paper-query ablation experiments")
    parser.add_argument("--videos", default="drtest")
    parser.add_argument("--s", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--obj_nums", default="2,3,4")
    parser.add_argument("--frame_threshold", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed_types", default="")
    parser.add_argument("--allowed_types_by_obj_num", default="")
    parser.add_argument("--include_no_window_split", action="store_true")
    parser.add_argument("--out_dir", default="supplement/out/paper_ablation")
    args = parser.parse_args()

    videos = [token.strip() for token in args.videos.split(",") if token.strip()]
    obj_nums = [int(token.strip()) for token in args.obj_nums.split(",") if token.strip()]
    allowed_types = {
        int(token.strip()) for token in args.allowed_types.split(",") if token.strip()
    } or None
    allowed_types_by_obj_num = _parse_allowed_types_by_obj_num(args.allowed_types_by_obj_num)
    rng = random.Random(args.seed)

    raw_rows: list[AblationRow] = []
    for video in videos:
        index, windowid_index, cp_graphs = _load_artifacts_for_s(video, args.s)
        no_split_artifacts = None
        if args.include_no_window_split:
            _ensure_no_split_index(video)
            no_split_artifacts = _load_artifacts_for_tag(video, "nosplit")

        for obj_num in obj_nums:
            sample_index = _filter_index_by_allowed_types(
                index,
                allowed_types_by_obj_num.get(obj_num, allowed_types),
            )
            for _ in range(args.n):
                query_types = sample_query_types(sample_index, obj_num, rng)
                theta_lt, d_lt = sample_thresholds_from_cp_graphs(cp_graphs, rng)

                reference_run = run_query_variant(
                    index=index,
                    cp_graphs=cp_graphs,
                    windowid_index=windowid_index,
                    query_object_types=query_types,
                    query_spatial=(theta_lt, d_lt),
                    topk=args.topk,
                    frame_threshold=args.frame_threshold,
                    variant="edge_exact",
                )
                reference_units = result_frame_units(reference_run.raw_results)

                variants = [
                    ("full_mli", ("full", index, windowid_index, cp_graphs)),
                    ("w_o_predecessor", ("current_vertex_only", index, windowid_index, cp_graphs)),
                    ("w_o_compressed_graph_query", ("frame_scan_exact", index, windowid_index, cp_graphs)),
                    ("w_o_cross_window_merge", ("no_cross_window_merge", index, windowid_index, cp_graphs)),
                ]
                if no_split_artifacts is not None:
                    ns_index, ns_windowid_index, ns_cp_graphs = no_split_artifacts
                    variants.append(
                        ("w_o_window_split", ("full", ns_index, ns_windowid_index, ns_cp_graphs))
                    )

                for label, (variant_key, active_index, active_windowid_index, active_cp_graphs) in variants:
                    run = run_query_variant(
                        index=active_index,
                        cp_graphs=active_cp_graphs,
                        windowid_index=active_windowid_index,
                        query_object_types=query_types,
                        query_spatial=(theta_lt, d_lt),
                        topk=args.topk,
                        frame_threshold=args.frame_threshold,
                        variant=variant_key,
                    )
                    metrics = compute_set_metrics(reference_units, result_frame_units(run.raw_results))
                    raw_rows.append(
                        AblationRow(
                            video=video,
                            s=str(args.s),
                            obj_num=obj_num,
                            query_types=",".join(map(str, query_types)),
                            theta_lt=theta_lt,
                            d_lt=d_lt,
                            variant=label,
                            frame_threshold=args.frame_threshold,
                            topk=args.topk,
                            windows_total=run.stats.windows_total,
                            windows_after_type_filter=run.stats.windows_after_type_filter,
                            candidate_vertices_total=run.stats.candidate_vertices_total,
                            bindings_total=run.stats.bindings_total,
                            bindings_matched=run.stats.bindings_matched,
                            pair_lookups=run.stats.pair_lookups,
                            theta_d_bins_scanned=run.stats.theta_d_bins_scanned,
                            t_type_filter_s=run.stage_times.type_filter_s,
                            t_candidate_vertex_s=run.stage_times.candidate_vertex_s,
                            t_spatial_match_s=run.stage_times.spatial_match_s,
                            t_prefix_tree_s=run.stage_times.prefix_tree_s,
                            t_total_s=run.stage_times.total_s,
                            result_unit_count=len(result_frame_units(run.raw_results)),
                            top_path_count=len(run.top_paths),
                            top1_frame_count=run.top_paths[0][2] if run.top_paths else 0,
                            precision=metrics["precision"],
                            recall=metrics["recall"],
                            f1=metrics["f1"],
                            jaccard=metrics["jaccard"],
                        )
                    )

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "paper_ablation_raw.csv"
    summary_path = out_dir / "paper_ablation_summary.csv"
    _write_csv(raw_path, [asdict(row) for row in raw_rows])
    _write_csv(summary_path, _summarize_rows(raw_rows))

    print(f"Wrote ablation rows: {raw_path}")
    print(f"Wrote ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
