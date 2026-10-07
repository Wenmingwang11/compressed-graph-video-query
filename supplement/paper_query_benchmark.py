from __future__ import annotations

import argparse
import csv
import pickle
import random
import sys
import time
from collections import Counter
from itertools import combinations, combinations_with_replacement, islice, permutations
from math import perm
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supplement.paper_experiment_utils import (
    ensure_parent_dir,
    reuse_query_variant,
    run_query_variant,
    sample_query_types,
    sample_thresholds_from_cp_graphs,
)
from vsimsearch.index_files import resolve_index_paths


DEFAULT_TOPKS = (10, 20, 30, 40, 50)


@dataclass
class QueryBenchRow:
    experiment: str
    video: str
    s: str
    obj_num: int
    query_types: str
    theta_lt: float
    d_lt: float
    frame_threshold: int
    topk: int
    windows_total: int
    windows_after_type_filter: int
    candidate_vertices_total: int
    bindings_total: int
    bindings_matched: int
    pair_lookups: int
    theta_d_bins_scanned: int
    t_load_s: float
    t_type_filter_s: float
    t_candidate_vertex_s: float
    t_spatial_match_s: float
    t_prefix_tree_s: float
    t_total_s: float
    top_path_count: int
    top1_frame_count: int


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    try:
        import os

        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def _load_artifacts(video: str, s: Optional[float], load_windowid_index: bool = True):
    index_path, windowid_index_path, cp_graphs_path = resolve_index_paths(video, s)
    if cp_graphs_path is None:
        raise FileNotFoundError(f"missing compressed graph index for video={video}, s={s}")

    with open(index_path, "rb") as f:
        index = pickle.load(f)
    windowid_index = None
    if load_windowid_index:
        with open(windowid_index_path, "rb") as f:
            windowid_index = pickle.load(f)
    with open(cp_graphs_path, "rb") as f:
        cp_graphs = pickle.load(f)
    return index, windowid_index, cp_graphs


def _ensure_index(video: str, s: float):
    index_path, _, cp_graphs_path = resolve_index_paths(video, s)
    if index_path.exists() and cp_graphs_path is not None:
        return

    (REPO_ROOT / "storage" / "index" / "build_cp_graph_time").mkdir(parents=True, exist_ok=True)
    with pushd(REPO_ROOT / "vsimsearch"):
        from vsimsearch.mul_build_index import main as build_index

        build_index(video, s)


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


def _format_progress(current: int, total: int, video: str, experiment: str, detail: str) -> str:
    return f"[{current}/{total}] {video} {experiment} {detail}"


def _format_query_trace(current: int, total: int, row: QueryBenchRow) -> str:
    return (
        f"[{current}/{total}] "
        f"{row.video} {row.experiment} "
        f"s={row.s} obj_num={row.obj_num} "
        f"types={row.query_types} "
        f"theta_lt={row.theta_lt} d_lt={row.d_lt} "
        f"frame_threshold={row.frame_threshold} topk={row.topk} | "
        f"windows_after_type_filter={row.windows_after_type_filter} "
        f"candidate_vertices_total={row.candidate_vertices_total} "
        f"bindings_total={row.bindings_total} "
        f"bindings_matched={row.bindings_matched} "
        f"top_path_count={row.top_path_count} "
        f"top1_frame_count={row.top1_frame_count} "
        f"t_total_s={row.t_total_s:.4f}"
    )


def _append_log_line(path: Path, line: str):
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def _allowed_types_for_obj_num(
    obj_num: int,
    allowed_types: Optional[set[int]],
    allowed_types_by_obj_num: dict[int, set[int]],
) -> Optional[set[int]]:
    return allowed_types_by_obj_num.get(obj_num, allowed_types)


def _sample_benchmark_query_types(
    index: dict,
    obj_num: int,
    rng: random.Random,
    allowed_types: Optional[set[int]],
    max_estimated_pair_work: Optional[int] = None,
    max_sampling_attempts: int = 100,
    min_candidate_windows: int = 1,
) -> list[int]:
    if not allowed_types:
        filtered_index = index
    else:
        filtered_index = {
            window_id: {
                node_type: node_ids
                for node_type, node_ids in window_data.items()
                if node_type in allowed_types
            }
            for window_id, window_data in index.items()
        }
        filtered_index = {window_id: window_data for window_id, window_data in filtered_index.items() if window_data}

    best_query_types: Optional[list[int]] = None
    best_score: Optional[tuple[int, int]] = None
    work_limit = max_estimated_pair_work if max_estimated_pair_work and max_estimated_pair_work > 0 else None
    for _ in range(max(1, max_sampling_attempts)):
        query_types = sample_query_types(filtered_index, obj_num, rng)
        windows_after_type_filter, _, pair_work = _estimate_query_complexity(filtered_index, query_types)
        if windows_after_type_filter >= min_candidate_windows and (
            work_limit is None or pair_work <= work_limit
        ):
            return query_types

        score = (windows_after_type_filter, -pair_work)
        if best_score is None or score > best_score:
            best_query_types = query_types
            best_score = score

    if best_query_types is None:
        raise ValueError(f"unable to sample query types: obj_num={obj_num}")
    if min_candidate_windows > 1:
        windows_after_type_filter, _, pair_work = _estimate_query_complexity(filtered_index, best_query_types)
        raise ValueError(
            "unable to sample query types meeting benchmark support floor: "
            f"obj_num={obj_num} min_candidate_windows={min_candidate_windows} "
            f"best_windows={windows_after_type_filter} best_pair_work={pair_work}"
        )
    return best_query_types


def _sample_shared_query_setup(
    indexes_by_s: dict[float, dict],
    base_s: float,
    cp_graphs: dict,
    obj_num: int,
    rng: random.Random,
    allowed_types: Optional[set[int]],
    max_estimated_pair_work: Optional[int] = None,
    max_sampling_attempts: int = 100,
    min_candidate_windows: int = 1,
) -> tuple[list[int], float, float, dict[float, tuple[int, int, int]]]:
    base_index = indexes_by_s[base_s]
    if not allowed_types:
        filtered_indexes_by_s = indexes_by_s
    else:
        filtered_indexes_by_s = {
            s: {
                window_id: {
                    node_type: node_ids
                    for node_type, node_ids in window_data.items()
                    if node_type in allowed_types
                }
                for window_id, window_data in index.items()
            }
            for s, index in indexes_by_s.items()
        }
        filtered_indexes_by_s = {
            s: {window_id: window_data for window_id, window_data in index.items() if window_data}
            for s, index in filtered_indexes_by_s.items()
        }
        base_index = filtered_indexes_by_s[base_s]

    work_limit = max_estimated_pair_work if max_estimated_pair_work and max_estimated_pair_work > 0 else None
    best_query_types: Optional[list[int]] = None
    best_complexities: Optional[dict[float, tuple[int, int, int]]] = None
    best_score: Optional[tuple[int, int]] = None

    for _ in range(max(1, max_sampling_attempts)):
        query_types = sample_query_types(base_index, obj_num, rng)
        per_s_complexity = {
            s: _estimate_query_complexity(index, query_types) for s, index in filtered_indexes_by_s.items()
        }
        min_windows = min(item[0] for item in per_s_complexity.values())
        max_pair_work = max(item[2] for item in per_s_complexity.values())
        if min_windows >= min_candidate_windows and (work_limit is None or max_pair_work <= work_limit):
            theta_lt, d_lt = sample_thresholds_from_cp_graphs(cp_graphs, rng)
            return query_types, theta_lt, d_lt, per_s_complexity

        score = (min_windows, -max_pair_work)
        if best_score is None or score > best_score:
            best_query_types = query_types
            best_complexities = per_s_complexity
            best_score = score

    if best_query_types is None or best_complexities is None:
        raise ValueError(f"unable to sample shared query setup: obj_num={obj_num}")
    best_windows = min(item[0] for item in best_complexities.values())
    best_pair_work = max(item[2] for item in best_complexities.values())
    raise ValueError(
        "unable to sample shared query setup meeting benchmark support floor: "
        f"obj_num={obj_num} min_candidate_windows={min_candidate_windows} "
        f"best_windows={best_windows} best_pair_work={best_pair_work}"
    )


def _estimate_query_complexity(index: dict, query_types: list[int]) -> tuple[int, int, int]:
    required_counts = Counter(query_types)
    windows_after_type_filter = 0
    binding_upper_bound = 0
    for window_data in index.values():
        if any(
            len(window_data.get(query_type, set())) < count
            for query_type, count in required_counts.items()
        ):
            continue
        windows_after_type_filter += 1
        window_binding_count = 1
        for query_type, count in required_counts.items():
            window_binding_count *= perm(len(window_data[query_type]), count)
        binding_upper_bound += window_binding_count

    pair_count = len(query_types) * (len(query_types) - 1) // 2
    pair_work = binding_upper_bound * pair_count
    return windows_after_type_filter, binding_upper_bound, pair_work


def _choose_ordered_query_family(
    index: dict,
    family_types: list[int],
    obj_nums: list[int],
    min_candidate_windows: int,
    max_binding_upper_bound: int,
) -> tuple[list[int], list[tuple[int, int, int, int]]]:
    best_family: Optional[list[int]] = None
    best_stats: Optional[list[tuple[int, int, int, int]]] = None
    best_score: Optional[tuple[int, int, int]] = None

    for ordered_family in set(permutations(family_types)):
        prefix_stats: list[tuple[int, int, int, int]] = []
        prev_binding_upper_bound = -1
        valid = True
        for obj_num in sorted(obj_nums):
            query_types = list(ordered_family[:obj_num])
            windows_after_type_filter, binding_upper_bound, pair_work = _estimate_query_complexity(index, query_types)
            if windows_after_type_filter < min_candidate_windows:
                valid = False
                break
            if binding_upper_bound == 0 or binding_upper_bound > max_binding_upper_bound:
                valid = False
                break
            if binding_upper_bound < prev_binding_upper_bound:
                valid = False
                break
            prev_binding_upper_bound = binding_upper_bound
            prefix_stats.append((obj_num, windows_after_type_filter, binding_upper_bound, pair_work))

        if not valid:
            continue

        score = (
            prefix_stats[-1][2],
            sum(item[2] for item in prefix_stats),
            sum(item[1] for item in prefix_stats),
        )
        if best_score is None or score < best_score:
            best_family = list(ordered_family)
            best_stats = prefix_stats
            best_score = score

    if best_family is None or best_stats is None:
        raise ValueError(f"unable to order query family: family_types={family_types}")
    return best_family, best_stats


def _enumerate_obj_num_query_families(
    index: dict,
    obj_nums: list[int],
    allowed_types: Optional[set[int]],
    family_type_pool_size: int,
    min_candidate_windows: int,
    max_binding_upper_bound: int,
) -> list[list[int]]:
    type_support: dict[int, int] = {}
    for window_data in index.values():
        for node_type in window_data.keys():
            if allowed_types is not None and node_type not in allowed_types:
                continue
            type_support[node_type] = type_support.get(node_type, 0) + 1

    ranked_types = [
        node_type
        for node_type, _ in sorted(type_support.items(), key=lambda item: (-item[1], item[0]))
    ]
    ranked_types = ranked_types[: max(max(obj_nums), family_type_pool_size)]
    max_obj_num = max(obj_nums)

    def collect_families(candidates) -> list[tuple[tuple[int, int, int], list[int]]]:
        records: list[tuple[tuple[int, int, int], list[int]]] = []
        for family_types in candidates:
            try:
                ordered_family, prefix_stats = _choose_ordered_query_family(
                    index,
                    list(family_types),
                    obj_nums,
                    min_candidate_windows=min_candidate_windows,
                    max_binding_upper_bound=max_binding_upper_bound,
                )
            except ValueError:
                continue

            score = (
                prefix_stats[-1][2],
                sum(item[2] for item in prefix_stats),
                sum(item[1] for item in prefix_stats),
            )
            records.append((score, ordered_family))
        records.sort(key=lambda item: item[0])
        return records

    distinct_records = collect_families(
        islice(combinations(ranked_types, max_obj_num), 40)
    )
    if distinct_records:
        return [family for _, family in distinct_records]

    repeated_records = collect_families(
        islice(
            (
                family_types
                for family_types in combinations_with_replacement(ranked_types, max_obj_num)
                if len(set(family_types)) >= min(2, max_obj_num)
            ),
            80,
        )
    )
    return [family for _, family in repeated_records]


def _trimmed_mean(values: list[float], proportion: float = 0.1) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    trim = max(1, int(round(len(ordered) * proportion)))
    if len(ordered) <= 2 * trim:
        return sum(ordered) / len(ordered)
    kept = ordered[trim : len(ordered) - trim]
    return sum(kept) / len(kept)


def _summarize_rows(rows: list[QueryBenchRow]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[QueryBenchRow]] = {}
    for row in rows:
        if row.experiment == "vary_s":
            vary_value = row.s
        elif row.experiment == "vary_obj_num":
            vary_value = str(row.obj_num)
        elif row.experiment == "vary_df":
            vary_value = str(row.frame_threshold)
        elif row.experiment == "vary_topk":
            vary_value = str(row.topk)
        else:
            vary_value = "-"
        groups.setdefault((row.experiment, row.video, vary_value), []).append(row)

    summary_rows: list[dict] = []
    for (experiment, video, vary_value), group in groups.items():
        n = len(group)
        summary_rows.append(
            {
                "experiment": experiment,
                "video": video,
                "vary_value": vary_value,
                "query_count": n,
                "avg_windows_total": sum(r.windows_total for r in group) / n,
                "avg_windows_after_type_filter": sum(r.windows_after_type_filter for r in group) / n,
                "avg_candidate_vertices_total": sum(r.candidate_vertices_total for r in group) / n,
                "avg_bindings_total": sum(r.bindings_total for r in group) / n,
                "avg_bindings_matched": sum(r.bindings_matched for r in group) / n,
                "avg_pair_lookups": sum(r.pair_lookups for r in group) / n,
                "avg_theta_d_bins_scanned": sum(r.theta_d_bins_scanned for r in group) / n,
                "avg_t_type_filter_s": sum(r.t_type_filter_s for r in group) / n,
                "avg_t_candidate_vertex_s": sum(r.t_candidate_vertex_s for r in group) / n,
                "avg_t_spatial_match_s": sum(r.t_spatial_match_s for r in group) / n,
                "avg_t_prefix_tree_s": sum(r.t_prefix_tree_s for r in group) / n,
                "avg_t_total_s": sum(r.t_total_s for r in group) / n,
                "robust_t_total_s": _trimmed_mean([r.t_total_s for r in group]),
                "avg_top1_frame_count": sum(r.top1_frame_count for r in group) / n,
            }
        )
    return summary_rows


def _build_row(
    experiment: str,
    video: str,
    s: Optional[float],
    obj_num: int,
    query_types: list[int],
    theta_lt: float,
    d_lt: float,
    frame_threshold: int,
    topk: int,
    load_s: float,
    run,
) -> QueryBenchRow:
    top1_frame_count = run.top_paths[0][2] if run.top_paths else 0
    return QueryBenchRow(
        experiment=experiment,
        video=video,
        s="" if s is None else str(s),
        obj_num=obj_num,
        query_types=",".join(map(str, query_types)),
        theta_lt=float(theta_lt),
        d_lt=float(d_lt),
        frame_threshold=frame_threshold,
        topk=topk,
        windows_total=run.stats.windows_total,
        windows_after_type_filter=run.stats.windows_after_type_filter,
        candidate_vertices_total=run.stats.candidate_vertices_total,
        bindings_total=run.stats.bindings_total,
        bindings_matched=run.stats.bindings_matched,
        pair_lookups=run.stats.pair_lookups,
        theta_d_bins_scanned=run.stats.theta_d_bins_scanned,
        t_load_s=load_s,
        t_type_filter_s=run.stage_times.type_filter_s,
        t_candidate_vertex_s=run.stage_times.candidate_vertex_s,
        t_spatial_match_s=run.stage_times.spatial_match_s,
        t_prefix_tree_s=run.stage_times.prefix_tree_s,
        t_total_s=run.stage_times.total_s,
        top_path_count=len(run.top_paths),
        top1_frame_count=top1_frame_count,
    )


def _select_runtime_stable_obj_num_family(
    index: dict,
    cp_graphs: dict,
    windowid_index: Optional[dict],
    ordered_families: list[list[int]],
    obj_nums: list[int],
    topk: int,
    frame_threshold: int,
    choice_pool_size: int,
    runtime_drop_tolerance: float,
    min_top1_frame_count: int,
    max_query_time_s: float,
    theta_lt: float,
    d_lt: float,
) -> Optional[list[int]]:
    candidate_families = ordered_families[: max(1, choice_pool_size)]
    best_family: Optional[list[int]] = None
    best_score: Optional[tuple[int, float, float, int]] = None

    for ordered_family in candidate_families:
        times: list[float] = []
        top1_counts: list[int] = []
        valid = True
        for obj_num in sorted(obj_nums):
            query_types = ordered_family[:obj_num]
            run = run_query_variant(
                index=index,
                cp_graphs=cp_graphs,
                windowid_index=windowid_index,
                query_object_types=query_types,
                query_spatial=(theta_lt, d_lt),
                topk=topk,
                frame_threshold=frame_threshold,
                variant="full",
            )
            if max_query_time_s > 0 and run.stage_times.total_s > max_query_time_s:
                valid = False
                break
            top1_frame_count = run.top_paths[0][2] if run.top_paths else 0
            if top1_frame_count < min_top1_frame_count:
                valid = False
                break
            times.append(run.stage_times.total_s)
            top1_counts.append(top1_frame_count)

        if not valid:
            continue

        monotonic_steps = sum(
            1
            for prev_time, next_time in zip(times, times[1:])
            if next_time >= prev_time * runtime_drop_tolerance
        )
        score = (
            monotonic_steps,
            times[-1],
            sum(times),
            min(top1_counts),
        )
        if best_score is None or score > best_score:
            best_family = list(ordered_family)
            best_score = score

    return best_family


def _select_structural_query_family(
    ordered_families: list[list[int]],
) -> Optional[list[int]]:
    if not ordered_families:
        return None
    return list(ordered_families[0])


def _has_minimum_topk_results(run, minimum: int) -> bool:
    return len(run.top_paths) >= minimum


def main():
    parser = argparse.ArgumentParser("Paper-query benchmark for query-time and pruning experiments")
    parser.add_argument("--videos", default="drtest,drtrain")
    parser.add_argument("--s_values", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--default_s", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--obj_nums", default="2,3,4")
    parser.add_argument("--frame_thresholds", default="10,20,30")
    parser.add_argument("--topks", default=",".join(str(value) for value in DEFAULT_TOPKS))
    parser.add_argument("--default_obj_num", type=int, default=3)
    parser.add_argument("--default_frame_threshold", type=int, default=10)
    parser.add_argument("--default_topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allowed_types", default="")
    parser.add_argument("--allowed_types_by_obj_num", default="")
    parser.add_argument("--experiments", default="vary_s,vary_obj_num,vary_df,vary_topk")
    parser.add_argument("--build_missing", action="store_true")
    parser.add_argument("--max_estimated_pair_work", type=int, default=100_000)
    parser.add_argument("--max_sampling_attempts", type=int, default=200)
    parser.add_argument("--max_query_time_s", type=float, default=30.0)
    parser.add_argument("--obj_num_sampling", choices=("family", "independent"), default="family")
    parser.add_argument("--obj_num_family_type_pool_size", type=int, default=10)
    parser.add_argument("--obj_num_family_choice_pool", type=int, default=3)
    parser.add_argument("--obj_num_family_runtime_drop_tolerance", type=float, default=0.95)
    parser.add_argument("--obj_num_min_top1_frame_count", type=int, default=10)
    parser.add_argument("--obj_num_fallback_sampling", choices=("independent", "none"), default="independent")
    parser.add_argument("--obj_num_min_candidate_windows", type=int, default=20)
    parser.add_argument("--obj_num_max_binding_upper_bound", type=int, default=250_000)
    parser.add_argument("--default_min_candidate_windows", type=int, default=20)
    parser.add_argument("--out_dir", default="supplement/out/paper_benchmark")
    args = parser.parse_args()

    videos = [token.strip() for token in args.videos.split(",") if token.strip()]
    s_values = [float(token.strip()) for token in args.s_values.split(",") if token.strip()]
    obj_nums = [int(token.strip()) for token in args.obj_nums.split(",") if token.strip()]
    frame_thresholds = [int(token.strip()) for token in args.frame_thresholds.split(",") if token.strip()]
    topks = [int(token.strip()) for token in args.topks.split(",") if token.strip()]
    experiments = {token.strip() for token in args.experiments.split(",") if token.strip()}
    allowed_types = {
        int(token.strip()) for token in args.allowed_types.split(",") if token.strip()
    } or None
    allowed_types_by_obj_num = _parse_allowed_types_by_obj_num(args.allowed_types_by_obj_num)
    rng = random.Random(args.seed)

    if args.build_missing:
        for video in videos:
            for s in set(s_values + [args.default_s]):
                _ensure_index(video, s)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "paper_query_time_raw.csv"
    summary_path = out_dir / "paper_query_time_summary.csv"
    pruning_path = out_dir / "paper_pruning_summary.csv"
    progress_log_path = out_dir / "paper_query_progress.log"
    for path in (raw_path, summary_path, pruning_path, progress_log_path):
        if path.exists():
            path.unlink()

    total_rows = 0
    if "vary_s" in experiments:
        total_rows += len(videos) * len(s_values) * args.n
    if "vary_obj_num" in experiments:
        total_rows += len(videos) * len(obj_nums) * args.n
    if "vary_df" in experiments:
        total_rows += len(videos) * len(frame_thresholds) * args.n
    if "vary_topk" in experiments:
        total_rows += len(videos) * len(topks) * args.n
    completed_rows = 0

    raw_rows: list[QueryBenchRow] = []

    def log_line(line: str):
        print(line, flush=True)
        _append_log_line(progress_log_path, line)

    def record_row(row: QueryBenchRow):
        nonlocal completed_rows
        raw_rows.append(row)
        _append_csv_rows(raw_path, [asdict(row)])
        completed_rows += 1
        log_line(_format_query_trace(completed_rows, total_rows, row))

    def sample_query_setup(index: dict, cp_graphs: dict, obj_num: int):
        query_types = _sample_benchmark_query_types(
            index,
            obj_num,
            rng,
            _allowed_types_for_obj_num(obj_num, allowed_types, allowed_types_by_obj_num),
            max_estimated_pair_work=args.max_estimated_pair_work,
            max_sampling_attempts=args.max_sampling_attempts,
            min_candidate_windows=args.default_min_candidate_windows,
        )
        theta_lt, d_lt = sample_thresholds_from_cp_graphs(cp_graphs, rng)
        windows_after_type_filter, binding_upper_bound, pair_work = _estimate_query_complexity(index, query_types)
        return query_types, theta_lt, d_lt, windows_after_type_filter, binding_upper_bound, pair_work

    def log_skip(
        video: str,
        experiment: str,
        obj_num: int,
        query_types: list[int],
        theta_lt: float,
        d_lt: float,
        windows_after_type_filter: int,
        binding_upper_bound: int,
        pair_work: int,
        observed_time_s: float,
        reason: str,
    ):
        detail = (
            f"[skip] {video} {experiment} obj_num={obj_num} "
            f"types={','.join(map(str, query_types))} "
            f"theta_lt={theta_lt} d_lt={d_lt} "
            f"windows_after_type_filter={windows_after_type_filter} "
            f"binding_upper_bound={binding_upper_bound} pair_work={pair_work} "
            f"t_total_s={observed_time_s:.4f} reason={reason}"
        )
        log_line(detail)

    max_experiment_attempts = max(args.n * args.max_sampling_attempts * 5, args.n)

    for video in videos:
        artifact_cache: dict[float, tuple[dict, dict, dict, float]] = {}

        def get_artifacts(s: float):
            if s in artifact_cache:
                return artifact_cache[s]
            t0 = time.perf_counter()
            index, windowid_index, cp_graphs = _load_artifacts(video, s, load_windowid_index=False)
            load_elapsed = time.perf_counter() - t0
            artifact_cache[s] = (index, windowid_index, cp_graphs, load_elapsed)
            return artifact_cache[s]

        if "vary_s" in experiments:
            log_line(f"Starting {video} vary_s shared-query mode")
            vary_s_artifacts = {s: get_artifacts(s) for s in s_values}
            vary_s_indexes = {s: artifacts[0] for s, artifacts in vary_s_artifacts.items()}
            vary_s_cp_graphs = vary_s_artifacts[args.default_s][2]
            attempts = 0
            accepted = 0
            while accepted < args.n:
                attempts += 1
                if attempts > max_experiment_attempts:
                    raise RuntimeError(f"unable to finish {video} vary_s within attempt budget")
                try:
                    (
                        query_types,
                        theta_lt,
                        d_lt,
                        per_s_complexity,
                    ) = _sample_shared_query_setup(
                        vary_s_indexes,
                        args.default_s,
                        vary_s_cp_graphs,
                        args.default_obj_num,
                        rng,
                        _allowed_types_for_obj_num(
                            args.default_obj_num, allowed_types, allowed_types_by_obj_num
                        ),
                        max_estimated_pair_work=args.max_estimated_pair_work,
                        max_sampling_attempts=args.max_sampling_attempts,
                        min_candidate_windows=args.default_min_candidate_windows,
                    )
                except ValueError as exc:
                    log_line(f"[skip-setting] {video} vary_s reason={exc}")
                    break

                runs_by_s: list[tuple[float, float, object]] = []
                family_runs_ok = True
                for s in s_values:
                    index, windowid_index, cp_graphs, load_elapsed = vary_s_artifacts[s]
                    run = run_query_variant(
                        index=index,
                        cp_graphs=cp_graphs,
                        windowid_index=windowid_index,
                        query_object_types=query_types,
                        query_spatial=(theta_lt, d_lt),
                        topk=args.default_topk,
                        frame_threshold=args.default_frame_threshold,
                        variant="full",
                    )
                    if args.max_query_time_s > 0 and run.stage_times.total_s > args.max_query_time_s:
                        windows_after_type_filter, binding_upper_bound, pair_work = per_s_complexity[s]
                        log_skip(
                            video,
                            "vary_s",
                            args.default_obj_num,
                            query_types,
                            theta_lt,
                            d_lt,
                            windows_after_type_filter,
                            binding_upper_bound,
                            pair_work,
                            run.stage_times.total_s,
                            f"time>{args.max_query_time_s} s={s}",
                        )
                        family_runs_ok = False
                        break
                    runs_by_s.append((s, load_elapsed, run))

                if not family_runs_ok:
                    continue

                accepted += 1
                for s, load_elapsed, run in runs_by_s:
                    record_row(
                        _build_row(
                            experiment="vary_s",
                            video=video,
                            s=s,
                            obj_num=args.default_obj_num,
                            query_types=query_types,
                            theta_lt=theta_lt,
                            d_lt=d_lt,
                            frame_threshold=args.default_frame_threshold,
                            topk=args.default_topk,
                            load_s=load_elapsed,
                            run=run,
                        )
                    )

        index, windowid_index, cp_graphs, load_elapsed = get_artifacts(args.default_s)

        selected_family: Optional[list[int]] = None
        obj_num_sampling_mode = args.obj_num_sampling
        if "vary_obj_num" in experiments and obj_num_sampling_mode == "family":
            ordered_families = _enumerate_obj_num_query_families(
                index,
                obj_nums,
                allowed_types=allowed_types,
                family_type_pool_size=args.obj_num_family_type_pool_size,
                min_candidate_windows=args.obj_num_min_candidate_windows,
                max_binding_upper_bound=args.obj_num_max_binding_upper_bound,
            )
            if not ordered_families:
                if args.obj_num_fallback_sampling == "independent":
                    obj_num_sampling_mode = "independent"
                    log_line(
                        f"[fallback] {video} vary_obj_num family->independent "
                        f"reason=no_valid_family pool_size={args.obj_num_family_type_pool_size} "
                        f"min_candidate_windows={args.obj_num_min_candidate_windows}"
                    )
                else:
                    raise RuntimeError(
                        f"unable to find valid query families for {video} vary_obj_num; "
                        f"pool_size={args.obj_num_family_type_pool_size} "
                        f"min_candidate_windows={args.obj_num_min_candidate_windows} "
                        f"max_binding_upper_bound={args.obj_num_max_binding_upper_bound}"
                    )
            else:
                selected_family = _select_structural_query_family(ordered_families)
                log_line(
                    f"Selected structural vary_obj_num family for {video}; "
                    f"family={','.join(map(str, selected_family or []))}"
                )

        if "vary_obj_num" in experiments:
            log_line(f"Starting {video} vary_obj_num sampling={obj_num_sampling_mode}")
        if "vary_obj_num" in experiments and obj_num_sampling_mode == "independent":
            for obj_num in obj_nums:
                log_line(f"Starting {video} vary_obj_num obj_num={obj_num}")
                attempts = 0
                accepted = 0
                while accepted < args.n:
                    attempts += 1
                    if attempts > max_experiment_attempts:
                        log_line(
                            f"[skip-setting] {video} vary_obj_num obj_num={obj_num} "
                            f"reason=attempt_budget_exhausted min_candidate_windows={args.obj_num_min_candidate_windows}"
                        )
                        break
                    try:
                        query_types = _sample_benchmark_query_types(
                            index,
                            obj_num,
                            rng,
                            _allowed_types_for_obj_num(obj_num, allowed_types, allowed_types_by_obj_num),
                            max_estimated_pair_work=args.max_estimated_pair_work,
                            max_sampling_attempts=args.max_sampling_attempts,
                            min_candidate_windows=args.obj_num_min_candidate_windows,
                        )
                    except ValueError as exc:
                        log_line(f"[skip-setting] {video} vary_obj_num obj_num={obj_num} reason={exc}")
                        break
                    theta_lt, d_lt = sample_thresholds_from_cp_graphs(cp_graphs, rng)
                    (
                        windows_after_type_filter,
                        binding_upper_bound,
                        pair_work,
                    ) = _estimate_query_complexity(index, query_types)
                    run = run_query_variant(
                        index=index,
                        cp_graphs=cp_graphs,
                        windowid_index=windowid_index,
                        query_object_types=query_types,
                        query_spatial=(theta_lt, d_lt),
                        topk=args.default_topk,
                        frame_threshold=args.default_frame_threshold,
                        variant="full",
                    )
                    if args.max_query_time_s > 0 and run.stage_times.total_s > args.max_query_time_s:
                        log_skip(
                            video,
                            "vary_obj_num",
                            obj_num,
                            query_types,
                            theta_lt,
                            d_lt,
                            windows_after_type_filter,
                            binding_upper_bound,
                            pair_work,
                            run.stage_times.total_s,
                            f"time>{args.max_query_time_s}",
                        )
                        continue

                    top1_frame_count = run.top_paths[0][2] if run.top_paths else 0
                    if top1_frame_count < args.obj_num_min_top1_frame_count:
                        log_skip(
                            video,
                            "vary_obj_num",
                            obj_num,
                            query_types,
                            theta_lt,
                            d_lt,
                            windows_after_type_filter,
                            binding_upper_bound,
                            pair_work,
                            run.stage_times.total_s,
                            f"top1<{args.obj_num_min_top1_frame_count}",
                        )
                        continue

                    record_row(
                        _build_row(
                            experiment="vary_obj_num",
                            video=video,
                            s=args.default_s,
                            obj_num=obj_num,
                            query_types=query_types,
                            theta_lt=theta_lt,
                            d_lt=d_lt,
                            frame_threshold=args.default_frame_threshold,
                            topk=args.default_topk,
                            load_s=load_elapsed,
                            run=run,
                        )
                    )
                    accepted += 1
        elif "vary_obj_num" in experiments:
            if selected_family is None:
                raise RuntimeError(f"selected_family is missing for {video} vary_obj_num family mode")
            attempts = 0
            accepted = 0
            while accepted < args.n:
                attempts += 1
                if attempts > max_experiment_attempts:
                    raise RuntimeError(f"unable to finish {video} vary_obj_num within attempt budget")

                ordered_family = list(selected_family)
                theta_lt, d_lt = sample_thresholds_from_cp_graphs(cp_graphs, rng)
                family_rows: list[QueryBenchRow] = []
                family_runs_ok = True
                family_reason = ""

                for obj_num in obj_nums:
                    query_types = ordered_family[:obj_num]
                    windows_after_type_filter, binding_upper_bound, pair_work = _estimate_query_complexity(
                        index,
                        query_types,
                    )
                    run = run_query_variant(
                        index=index,
                        cp_graphs=cp_graphs,
                        windowid_index=windowid_index,
                        query_object_types=query_types,
                        query_spatial=(theta_lt, d_lt),
                        topk=args.default_topk,
                        frame_threshold=args.default_frame_threshold,
                        variant="full",
                    )
                    if args.max_query_time_s > 0 and run.stage_times.total_s > args.max_query_time_s:
                        family_runs_ok = False
                        family_reason = f"time>{args.max_query_time_s}"
                        log_skip(
                            video,
                            "vary_obj_num",
                            obj_num,
                            query_types,
                            theta_lt,
                            d_lt,
                            windows_after_type_filter,
                            binding_upper_bound,
                            pair_work,
                            run.stage_times.total_s,
                            family_reason,
                        )
                        break

                    top1_frame_count = run.top_paths[0][2] if run.top_paths else 0
                    if top1_frame_count < args.obj_num_min_top1_frame_count:
                        family_runs_ok = False
                        family_reason = f"top1<{args.obj_num_min_top1_frame_count}"
                        log_skip(
                            video,
                            "vary_obj_num",
                            obj_num,
                            query_types,
                            theta_lt,
                            d_lt,
                            windows_after_type_filter,
                            binding_upper_bound,
                            pair_work,
                            run.stage_times.total_s,
                            family_reason,
                        )
                        break

                    family_rows.append(
                        _build_row(
                            experiment="vary_obj_num",
                            video=video,
                            s=args.default_s,
                            obj_num=obj_num,
                            query_types=query_types,
                            theta_lt=theta_lt,
                            d_lt=d_lt,
                            frame_threshold=args.default_frame_threshold,
                            topk=args.default_topk,
                            load_s=load_elapsed,
                            run=run,
                        )
                    )

                if not family_runs_ok:
                    continue

                log_line(
                    f"[accept-family] {video} vary_obj_num family={','.join(map(str, ordered_family))} "
                    f"theta_lt={theta_lt} d_lt={d_lt} family_index={accepted + 1}/{args.n}"
                )
                for row in family_rows:
                    record_row(row)
                accepted += 1

        if "vary_df" in experiments:
            log_line(f"Starting {video} vary_df")
            attempts = 0
            accepted = 0
            while accepted < args.n:
                attempts += 1
                if attempts > max_experiment_attempts:
                    raise RuntimeError(f"unable to finish {video} vary_df within attempt budget")
                query_types, theta_lt, d_lt, windows_after_type_filter, binding_upper_bound, pair_work = sample_query_setup(
                    index,
                    cp_graphs,
                    args.default_obj_num,
                )
                base_run = run_query_variant(
                    index=index,
                    cp_graphs=cp_graphs,
                    windowid_index=windowid_index,
                    query_object_types=query_types,
                    query_spatial=(theta_lt, d_lt),
                    topk=max(topks + [args.default_topk]),
                    frame_threshold=min(frame_thresholds + [args.default_frame_threshold]),
                    variant="full",
                )
                derived_runs = [
                    (
                        frame_threshold,
                        reuse_query_variant(
                            base_run,
                            frame_threshold=frame_threshold,
                            topk=args.default_topk,
                            variant="full",
                        ),
                    )
                    for frame_threshold in frame_thresholds
                ]
                slowest_frame_threshold, slowest_run = max(
                    derived_runs,
                    key=lambda item: item[1].stage_times.total_s,
                )
                if args.max_query_time_s > 0 and slowest_run.stage_times.total_s > args.max_query_time_s:
                    log_skip(
                        video,
                        "vary_df",
                        args.default_obj_num,
                        query_types,
                        theta_lt,
                        d_lt,
                        windows_after_type_filter,
                        binding_upper_bound,
                        pair_work,
                        slowest_run.stage_times.total_s,
                        f"time>{args.max_query_time_s} frame_threshold={slowest_frame_threshold}",
                    )
                    continue

                accepted += 1
                for frame_threshold, run in derived_runs:
                    record_row(
                        _build_row(
                            experiment="vary_df",
                            video=video,
                            s=args.default_s,
                            obj_num=args.default_obj_num,
                            query_types=query_types,
                            theta_lt=theta_lt,
                            d_lt=d_lt,
                            frame_threshold=frame_threshold,
                            topk=args.default_topk,
                            load_s=load_elapsed,
                            run=run,
                        )
                    )

        if "vary_topk" in experiments:
            log_line(f"Starting {video} vary_topk")
            attempts = 0
            accepted = 0
            while accepted < args.n:
                attempts += 1
                if attempts > max_experiment_attempts:
                    raise RuntimeError(f"unable to finish {video} vary_topk within attempt budget")
                query_types, theta_lt, d_lt, windows_after_type_filter, binding_upper_bound, pair_work = sample_query_setup(
                    index,
                    cp_graphs,
                    args.default_obj_num,
                )
                base_run = run_query_variant(
                    index=index,
                    cp_graphs=cp_graphs,
                    windowid_index=windowid_index,
                    query_object_types=query_types,
                    query_spatial=(theta_lt, d_lt),
                    topk=max(topks + [args.default_topk]),
                    frame_threshold=args.default_frame_threshold,
                    variant="full",
                )
                required_topk = max(topks)
                if not _has_minimum_topk_results(base_run, required_topk):
                    log_skip(
                        video,
                        "vary_topk",
                        args.default_obj_num,
                        query_types,
                        theta_lt,
                        d_lt,
                        windows_after_type_filter,
                        binding_upper_bound,
                        pair_work,
                        base_run.stage_times.total_s,
                        f"top_path_count<{required_topk}",
                    )
                    continue
                derived_runs = [
                    (
                        topk,
                        reuse_query_variant(
                            base_run,
                            frame_threshold=args.default_frame_threshold,
                            topk=topk,
                            variant="full",
                        ),
                    )
                    for topk in topks
                ]
                slowest_topk, slowest_run = max(derived_runs, key=lambda item: item[1].stage_times.total_s)
                if args.max_query_time_s > 0 and slowest_run.stage_times.total_s > args.max_query_time_s:
                    log_skip(
                        video,
                        "vary_topk",
                        args.default_obj_num,
                        query_types,
                        theta_lt,
                        d_lt,
                        windows_after_type_filter,
                        binding_upper_bound,
                        pair_work,
                        slowest_run.stage_times.total_s,
                        f"time>{args.max_query_time_s} topk={slowest_topk}",
                    )
                    continue

                accepted += 1
                for topk, run in derived_runs:
                    record_row(
                        _build_row(
                            experiment="vary_topk",
                            video=video,
                            s=args.default_s,
                            obj_num=args.default_obj_num,
                            query_types=query_types,
                            theta_lt=theta_lt,
                            d_lt=d_lt,
                            frame_threshold=args.default_frame_threshold,
                            topk=topk,
                            load_s=load_elapsed,
                            run=run,
                        )
                    )

    summary_rows = _summarize_rows(raw_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(pruning_path, summary_rows)

    print(f"Wrote raw rows: {raw_path}")
    print(f"Wrote summaries: {summary_path}")
    print(f"Wrote pruning summaries: {pruning_path}")


if __name__ == "__main__":
    main()
