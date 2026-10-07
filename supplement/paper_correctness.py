from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

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
class CorrectnessRow:
    video: str
    s: str
    obj_num: int
    query_types: str
    theta_lt: float
    d_lt: float
    frame_threshold: int
    topk: int
    ref_unit_count: int
    pred_unit_count: int
    top1_ref_frame_count: int
    top1_pred_frame_count: int
    exact_result_equal: int
    top1_equal: int
    precision: float
    recall: float
    f1: float
    jaccard: float


def _write_csv(path: Path, rows: list[dict]):
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _filter_index_by_allowed_types(index: dict, allowed_types: set[int] | None) -> dict:
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


def _load_artifacts(video: str, s: float):
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


def _summarize_rows(rows: list[CorrectnessRow]) -> list[dict]:
    groups: dict[tuple[str, int], list[CorrectnessRow]] = {}
    for row in rows:
        groups.setdefault((row.video, row.obj_num), []).append(row)

    summary_rows: list[dict] = []
    for (video, obj_num), group in groups.items():
        n = len(group)
        summary_rows.append(
            {
                "video": video,
                "obj_num": obj_num,
                "query_count": n,
                "avg_ref_unit_count": sum(r.ref_unit_count for r in group) / n,
                "avg_pred_unit_count": sum(r.pred_unit_count for r in group) / n,
                "exact_result_equal_ratio": sum(r.exact_result_equal for r in group) / n,
                "top1_equal_ratio": sum(r.top1_equal for r in group) / n,
                "avg_precision": sum(r.precision for r in group) / n,
                "avg_recall": sum(r.recall for r in group) / n,
                "avg_f1": sum(r.f1 for r in group) / n,
                "avg_jaccard": sum(r.jaccard for r in group) / n,
            }
        )
    return summary_rows


def main():
    parser = argparse.ArgumentParser("Paper-query correctness/completeness experiment")
    parser.add_argument("--videos", default="drtest,drtrain")
    parser.add_argument("--s", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--obj_nums", default="2,3,4")
    parser.add_argument("--frame_threshold", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed_types", default="")
    parser.add_argument("--allowed_types_by_obj_num", default="")
    parser.add_argument("--out_dir", default="supplement/out/paper_correctness")
    args = parser.parse_args()

    videos = [token.strip() for token in args.videos.split(",") if token.strip()]
    obj_nums = [int(token.strip()) for token in args.obj_nums.split(",") if token.strip()]
    allowed_types = {
        int(token.strip()) for token in args.allowed_types.split(",") if token.strip()
    } or None
    allowed_types_by_obj_num = _parse_allowed_types_by_obj_num(args.allowed_types_by_obj_num)
    rng = random.Random(args.seed)

    raw_rows: list[CorrectnessRow] = []
    for video in videos:
        index, windowid_index, cp_graphs = _load_artifacts(video, args.s)
        for obj_num in obj_nums:
            sample_index = _filter_index_by_allowed_types(
                index,
                allowed_types_by_obj_num.get(obj_num, allowed_types),
            )
            for _ in range(args.n):
                query_types = sample_query_types(sample_index, obj_num, rng)
                theta_lt, d_lt = sample_thresholds_from_cp_graphs(cp_graphs, rng)

                exact_run = run_query_variant(
                    index=index,
                    cp_graphs=cp_graphs,
                    windowid_index=windowid_index,
                    query_object_types=query_types,
                    query_spatial=(theta_lt, d_lt),
                    topk=args.topk,
                    frame_threshold=args.frame_threshold,
                    variant="edge_exact",
                )
                paper_run = run_query_variant(
                    index=index,
                    cp_graphs=cp_graphs,
                    windowid_index=windowid_index,
                    query_object_types=query_types,
                    query_spatial=(theta_lt, d_lt),
                    topk=args.topk,
                    frame_threshold=args.frame_threshold,
                    variant="full",
                )

                exact_units = result_frame_units(exact_run.raw_results)
                paper_units = result_frame_units(paper_run.raw_results)
                metrics = compute_set_metrics(exact_units, paper_units)

                exact_top1 = exact_run.top_paths[0][2] if exact_run.top_paths else 0
                paper_top1 = paper_run.top_paths[0][2] if paper_run.top_paths else 0
                raw_rows.append(
                    CorrectnessRow(
                        video=video,
                        s=str(args.s),
                        obj_num=obj_num,
                        query_types=",".join(map(str, query_types)),
                        theta_lt=theta_lt,
                        d_lt=d_lt,
                        frame_threshold=args.frame_threshold,
                        topk=args.topk,
                        ref_unit_count=len(exact_units),
                        pred_unit_count=len(paper_units),
                        top1_ref_frame_count=exact_top1,
                        top1_pred_frame_count=paper_top1,
                        exact_result_equal=int(exact_run.raw_results == paper_run.raw_results),
                        top1_equal=int(exact_top1 == paper_top1),
                        precision=metrics["precision"],
                        recall=metrics["recall"],
                        f1=metrics["f1"],
                        jaccard=metrics["jaccard"],
                    )
                )

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "paper_correctness_raw.csv"
    summary_path = out_dir / "paper_correctness_summary.csv"
    _write_csv(raw_path, [asdict(row) for row in raw_rows])
    _write_csv(summary_path, _summarize_rows(raw_rows))

    print(f"Wrote correctness rows: {raw_path}")
    print(f"Wrote correctness summary: {summary_path}")


if __name__ == "__main__":
    main()
