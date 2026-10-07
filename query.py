import argparse
import pickle
import time
from itertools import islice

from prefix_tree import build_tree_from_results, get_top_five_paths_with_threshold
from vsimsearch.index_files import resolve_index_paths
from vsimsearch.querying import (
    QueryStats,
    filter_windows_by_types,
    iter_node_combinations,
    match_combinations_in_window,
)


def run_single_query(
    video_name: str,
    s: float,
    query_types: list[int],
    query_spatial: tuple[float, float],
    topk: int,
    max_combos_per_window: int = 0,
):
    index_path, windowid_index_path, _ = resolve_index_paths(video_name, s)

    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    with open(windowid_index_path, 'rb') as f:
        windowid_index = pickle.load(f)

    stats = QueryStats(windows_total=len(index))
    start = time.time()

    filtered_index = filter_windows_by_types(index, query_types)
    stats.windows_after_type_filter = len(filtered_index)

    results: dict[int, dict[tuple[int, ...], set[int]]] = {}
    for window_id, window_data in filtered_index.items():
        combos = iter_node_combinations(window_data, query_types)
        if max_combos_per_window and max_combos_per_window > 0:
            combos = islice(combos, max_combos_per_window)
        results[window_id] = match_combinations_in_window(
            windowid_index[window_id],
            combos,
            query_spatial,
            stats=stats,
        )

    elapsed = time.time() - start
    tree_root = build_tree_from_results(results)

    threshold = 2
    top_paths = get_top_five_paths_with_threshold(tree_root, threshold, topk)
    return elapsed, stats, top_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run one structured query (type + theta/d thresholds)')
    parser.add_argument('--video', default='drtest')
    parser.add_argument('--s', type=float, default=0.6, help='window similarity threshold S used at indexing time')
    parser.add_argument('--types', type=str, default='0,1', help='comma-separated object types')
    parser.add_argument('--theta_lt', type=float, default=1.0)
    parser.add_argument('--d_lt', type=float, default=5.0)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max_combos_per_window', type=int, default=0, help='0 means no cap')
    args = parser.parse_args()

    query_types = [int(x) for x in args.types.split(',') if x.strip() != '']
    query_spatial = (args.theta_lt, args.d_lt)

    elapsed, stats, top_paths = run_single_query(
        args.video,
        args.s,
        query_types,
        query_spatial,
        args.topk,
        max_combos_per_window=args.max_combos_per_window,
    )

    print(
        f'elapsed={elapsed:.4f}s windows={stats.windows_after_type_filter}/{stats.windows_total} '
        f'combos={stats.combinations_total} matched={stats.combinations_matched} '
        f'pair_lookups={stats.pair_lookups} bins_scanned={stats.theta_d_bins_scanned}'
    )
    for path, frames, count in top_paths:
        print('Path:', '->'.join(map(str, path)), 'Frame Count:', count, 'Frames:', frames)
