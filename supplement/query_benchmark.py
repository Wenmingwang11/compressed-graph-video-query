from __future__ import annotations

import sys
import argparse
import csv
import pickle
import random
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path when running as a script: `python supplement/query_benchmark.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prefix_tree import build_tree_from_results, get_top_five_paths_with_threshold
from vsimsearch.index_files import resolve_index_paths
from vsimsearch.querying import (
    QueryStats,
    filter_windows_by_types,
    iter_node_combinations,
    match_combinations_in_window,
)

from supplement.metrics_utils import StageTimes, Timer


@dataclass
class QueryBenchRow:
    video: str
    s: str
    obj_num: int
    query_types: str
    theta_lt: float
    d_lt: float
    topk: int
    max_combos_per_window: int
    windows_total: int
    windows_after_type_filter: int
    combinations_total: int
    combinations_matched: int
    pair_lookups: int
    theta_d_bins_scanned: int
    t_load_s: float
    t_type_filter_s: float
    t_spatial_match_s: float
    t_prefix_tree_s: float
    t_total_s: float


def _sample_thresholds(windowid_index: dict, rng: random.Random) -> tuple[float, float]:
    # Try to sample some observed theta/d bins for a reasonable threshold.
    thetas: list[float] = []
    ds: list[float] = []
    for _, window_edge_index in islice(windowid_index.items(), 10):
        for _, theta_d_map in islice(window_edge_index.items(), 50):
            for (theta, d) in theta_d_map.keys():
                thetas.append(float(theta))
                ds.append(float(d))
                if len(thetas) >= 200:
                    break
            if len(thetas) >= 200:
                break
        if len(thetas) >= 200:
            break

    if not thetas or not ds:
        return 1.0, 5.0

    theta_lt = rng.choice(thetas)
    d_lt = rng.choice(ds)
    # Make it a bit less degenerate: raise threshold a little to include more matches.
    return float(theta_lt) + 1.0, float(d_lt) + 1.0


def _union_types(index: dict) -> list[int]:
    types = set()
    for window_data in index.values():
        types |= set(window_data.keys())
    return sorted(types)


def main():
    parser = argparse.ArgumentParser('Benchmark query runtime and candidate statistics (paper supplement)')
    parser.add_argument('--videos', type=str, default='drtest,drtrain')
    parser.add_argument('--s', type=str, default='0.6', help='comma-separated S values; use \"\" for legacy naming only')
    parser.add_argument('--n', type=int, default=20, help='queries per obj_num')
    parser.add_argument('--obj_nums', type=str, default='2,3,4')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--max_combos_per_window', type=int, default=2000, help='cap to avoid combinatorial explosion')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='supplement/out/query_benchmark.csv')
    args = parser.parse_args()

    videos = [v.strip() for v in args.videos.split(',') if v.strip()]
    s_values: list[Optional[float]] = []
    for token in args.s.split(','):
        token = token.strip()
        if token == '':
            s_values.append(None)
        else:
            s_values.append(float(token))

    obj_nums = [int(x.strip()) for x in args.obj_nums.split(',') if x.strip()]
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[QueryBenchRow] = []
    for video in videos:
        for s in s_values:
            index_path, windowid_index_path, _ = resolve_index_paths(video, s)

            times = StageTimes()
            with Timer() as t:
                with open(index_path, 'rb') as f:
                    index = pickle.load(f)
                with open(windowid_index_path, 'rb') as f:
                    windowid_index = pickle.load(f)
            times.load_s = t.elapsed()

            available_types = _union_types(index)
            if not available_types:
                continue

            for obj_num in obj_nums:
                if len(available_types) < obj_num:
                    continue

                for _ in range(args.n):
                    query_types = rng.sample(available_types, obj_num)
                    theta_lt, d_lt = _sample_thresholds(windowid_index, rng)

                    stats = QueryStats(windows_total=len(index))
                    stage = StageTimes(load_s=times.load_s)

                    with Timer() as t0:
                        filtered_index = filter_windows_by_types(index, query_types)
                    stage.type_filter_s = t0.elapsed()
                    stats.windows_after_type_filter = len(filtered_index)

                    results = {}
                    with Timer() as t1:
                        for window_id, window_data in filtered_index.items():
                            combos = iter_node_combinations(window_data, query_types)
                            combos = islice(combos, args.max_combos_per_window)
                            results[window_id] = match_combinations_in_window(
                                windowid_index[window_id],
                                combos,
                                (theta_lt, d_lt),
                                stats=stats,
                            )
                    stage.spatial_match_s = t1.elapsed()

                    with Timer() as t2:
                        tree_root = build_tree_from_results(results)
                        _ = get_top_five_paths_with_threshold(tree_root, threshold=2, key=args.topk)
                    stage.prefix_tree_s = t2.elapsed()

                    stage.total_s = stage.type_filter_s + stage.spatial_match_s + stage.prefix_tree_s

                    rows.append(
                        QueryBenchRow(
                            video=video,
                            s='' if s is None else str(s),
                            obj_num=obj_num,
                            query_types=','.join(map(str, query_types)),
                            theta_lt=float(theta_lt),
                            d_lt=float(d_lt),
                            topk=args.topk,
                            max_combos_per_window=args.max_combos_per_window,
                            windows_total=stats.windows_total,
                            windows_after_type_filter=stats.windows_after_type_filter,
                            combinations_total=stats.combinations_total,
                            combinations_matched=stats.combinations_matched,
                            pair_lookups=stats.pair_lookups,
                            theta_d_bins_scanned=stats.theta_d_bins_scanned,
                            t_load_s=stage.load_s,
                            t_type_filter_s=stage.type_filter_s,
                            t_spatial_match_s=stage.spatial_match_s,
                            t_prefix_tree_s=stage.prefix_tree_s,
                            t_total_s=stage.total_s,
                        )
                    )

    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f'Wrote {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    main()
