import os
import pickle
import time
from pathlib import Path

from prefix_tree import build_tree_from_results, get_top_five_paths_with_threshold
from vsimsearch.index_files import resolve_index_paths
from vsimsearch.querying import (
    QueryStats,
    filter_windows_by_types,
    iter_node_combinations,
    match_combinations_in_window,
)


def batch_query(query_objects_types, query_spatial, video_name, key, s, results_dir='./storage/results'):
    index_path, windowid_index_path, _ = resolve_index_paths(video_name, s)

    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    with open(windowid_index_path, 'rb') as f:
        windowid_index = pickle.load(f)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(results_dir, f'{video_name}_{s}_results.txt')

    total_time = 0.0
    for query_types in query_objects_types:
        stats = QueryStats(windows_total=len(index))
        start = time.time()

        filtered_index = filter_windows_by_types(index, query_types)
        stats.windows_after_type_filter = len(filtered_index)

        results = {}
        for window_id, window_data in filtered_index.items():
            combos = iter_node_combinations(window_data, query_types)
            results[window_id] = match_combinations_in_window(
                windowid_index[window_id],
                combos,
                query_spatial,
                stats=stats,
            )

        elapsed_time = time.time() - start
        total_time += elapsed_time

        tree_root = build_tree_from_results(results)
        threshold = 2
        top_paths = get_top_five_paths_with_threshold(tree_root, threshold, key)

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f'Query Types: {query_types}\n')
            f.write(f'Query Spatial (theta<, d<): {query_spatial}\n')
            f.write(f'Elapsed: {elapsed_time:.6f}s\n')
            f.write(
                f'Windows: {stats.windows_after_type_filter}/{stats.windows_total}, '
                f'Combos: {stats.combinations_total}, Matched: {stats.combinations_matched}, '
                f'PairLookups: {stats.pair_lookups}, BinsScanned: {stats.theta_d_bins_scanned}\n'
            )
            for path, frames, count in top_paths:
                f.write(
                    f"Path: {'->'.join(map(str, path))}, Frame Count: {count}, Frames: {sorted(frames)}\n"
                )
            f.write('\n')

    return total_time


if __name__ == '__main__':
    videos = ['drtest', 'drtrain']
    query_objects_types = [[5, 10], [3, 5, 10], [3, 5, 8, 10]]
    query_spatial = (10, 5)
    similiarity = [0.6]
    key = 10

    for v in videos:
        for s in similiarity:
            total = batch_query(query_objects_types, query_spatial, v, key, s)
            print(f'Total Time for {v} (S={s}): {total:.4f}s')
