from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _ensure_frame_ids_cache(video_name: str):
    """
    lists_cluster.py reads `storage/{video}.txt.frame-ids.pkl`.
    That file is produced by vsimsearch.io.read_nodes_from_mot_result_multi_class.
    """
    raw_path = REPO_ROOT / 'storage' / f'{video_name}.txt'
    frame_ids_path = REPO_ROOT / 'storage' / f'{video_name}.txt.frame-ids.pkl'
    if frame_ids_path.exists():
        return
    from vsimsearch.io import read_nodes_from_mot_result_multi_class

    print(f'[1/4] Creating frame-id cache: {frame_ids_path}')
    _ = read_nodes_from_mot_result_multi_class(str(raw_path), enable_cache=True)


def _ensure_subsection(video_name: str, s: float):
    subsection_path = REPO_ROOT / 'storage' / 'zhunbei' / f'{video_name}_{s}_subsection.pkl'
    if subsection_path.exists():
        return
    print(f'[2/4] Creating subsection file: {subsection_path}')
    from lists_cluster import main as build_subsection

    build_subsection(video_name, s)


def _ensure_index(video_name: str, s: float):
    index_path = REPO_ROOT / 'storage' / 'index' / f'{video_name}_{s}_index.pkl'
    windowid_index_path = REPO_ROOT / 'storage' / 'index' / f'{video_name}_{s}_windowid_index.pkl'
    if index_path.exists() and windowid_index_path.exists():
        return
    print(f'[3/4] Building index: {index_path.name}, {windowid_index_path.name}')

    # mul_build_index.py uses ../storage/... paths, so it must run with cwd=vsimsearch.
    with pushd(REPO_ROOT / 'vsimsearch'):
        from mul_build_index import main as build_index

        build_index(video_name, s)


def _run_query(
    video_name: str,
    s: float,
    types_csv: str,
    theta_lt: float,
    d_lt: float,
    topk: int,
    max_combos_per_window: int,
):
    from query import run_single_query

    query_types = [int(x) for x in types_csv.split(',') if x.strip() != '']
    query_spatial = (theta_lt, d_lt)

    return run_single_query(
        video_name=video_name,
        s=s,
        query_types=query_types,
        query_spatial=query_spatial,
        topk=topk,
        max_combos_per_window=max_combos_per_window,
    )


def _auto_choose_types(index: dict, obj_num: int, seed: int) -> list[int]:
    """
    Auto choose object types from the built window-type index.

    Heuristic:
    - Prefer type combinations that appear in many windows
    - Penalize combinations with large expected per-window Cartesian products
    """
    import random

    rng = random.Random(seed)
    if obj_num <= 0:
        raise ValueError('--obj_num must be >= 1')

    type_window_count: dict[int, int] = {}
    type_node_total: dict[int, int] = {}
    for window_data in index.values():
        for t, node_ids in window_data.items():
            type_window_count[t] = type_window_count.get(t, 0) + 1
            type_node_total[t] = type_node_total.get(t, 0) + len(node_ids)

    available = sorted(type_window_count.keys())
    if len(available) < obj_num:
        raise ValueError(f'not enough types in index: have={len(available)} need={obj_num}')

    ranked = sorted(
        available,
        key=lambda t: (type_window_count[t], -type_node_total[t]),
        reverse=True,
    )

    top_n = min(12, len(ranked))
    pool = ranked[:top_n]
    extras = [t for t in available if t not in pool]
    rng.shuffle(extras)
    pool.extend(extras[: min(6, len(extras))])
    pool = sorted(set(pool))

    if obj_num == 1:
        best = min(
            pool,
            key=lambda t: (
                -type_window_count[t],
                (type_node_total[t] / max(type_window_count[t], 1)),
                t,
            ),
        )
        return [best]

    best_combo: tuple[int, ...] | None = None
    best_score = -1.0
    for combo in combinations(pool, obj_num):
        required = set(combo)
        windows_with_all = 0
        product_sum = 0.0

        for window_data in index.values():
            if not required.issubset(window_data.keys()):
                continue
            windows_with_all += 1
            prod = 1
            for t in combo:
                prod *= len(window_data[t])
            product_sum += float(prod)

        if windows_with_all == 0:
            continue

        expected_prod = product_sum / windows_with_all
        score = windows_with_all / (expected_prod + 1e-9)
        if score > best_score:
            best_score = score
            best_combo = combo

    if best_combo is None:
        return rng.sample(available, obj_num)

    return list(best_combo)


def main():
    parser = argparse.ArgumentParser('One-click: cache -> subsection -> index -> query')
    parser.add_argument('--video', default='drtest', help='dataset name under storage/, e.g. drtest')
    parser.add_argument('--s', type=float, default=0.6, help='window similarity threshold S')
    parser.add_argument('--types', default='0,1', help='comma-separated object types, e.g. 0,1 or 3,5,10')
    parser.add_argument('--auto_types', action='store_true', help='auto choose object types from the built index')
    parser.add_argument('--obj_num', type=int, default=2, help='number of object types if --auto_types is set')
    parser.add_argument('--seed', type=int, default=42, help='random seed for --auto_types')
    parser.add_argument('--theta_lt', type=float, default=1.0)
    parser.add_argument('--d_lt', type=float, default=5.0)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--max_combos_per_window', type=int, default=2000)
    parser.add_argument('--skip_index', action='store_true', help='assume index exists, skip building')
    parser.add_argument('--skip_subsection', action='store_true', help='assume subsection exists, skip building')
    parser.add_argument('--skip_cache', action='store_true', help='assume frame-ids cache exists, skip building')
    parser.add_argument('--out_dir', default='supplement/out', help='output directory (gitignored)')
    args = parser.parse_args()

    raw_path = REPO_ROOT / 'storage' / f'{args.video}.txt'
    if not raw_path.exists():
        raise FileNotFoundError(f'missing raw file: {raw_path}')

    t0 = time.time()

    if not args.skip_cache:
        _ensure_frame_ids_cache(args.video)
    if not args.skip_subsection:
        _ensure_subsection(args.video, args.s)
    if not args.skip_index:
        _ensure_index(args.video, args.s)

    chosen_types_csv = args.types
    if args.auto_types:
        import pickle
        from vsimsearch.index_files import resolve_index_paths

        index_path, _, _ = resolve_index_paths(args.video, args.s)
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        chosen_types = _auto_choose_types(index, args.obj_num, args.seed)
        chosen_types_csv = ','.join(map(str, chosen_types))
        print(f'[auto_types] chosen types: {chosen_types_csv}')

    print('[4/4] Running query ...')
    elapsed, stats, top_paths = _run_query(
        video_name=args.video,
        s=args.s,
        types_csv=chosen_types_csv,
        theta_lt=args.theta_lt,
        d_lt=args.d_lt,
        topk=args.topk,
        max_combos_per_window=args.max_combos_per_window,
    )

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'one_query_{args.video}_S{args.s}_{ts}.json'

    payload = {
        'video': args.video,
        's': args.s,
        'types': chosen_types_csv,
        'auto_types': bool(args.auto_types),
        'obj_num': int(args.obj_num),
        'seed': int(args.seed),
        'theta_lt': args.theta_lt,
        'd_lt': args.d_lt,
        'topk': args.topk,
        'max_combos_per_window': args.max_combos_per_window,
        'query_elapsed_s': elapsed,
        'pipeline_elapsed_s': time.time() - t0,
        'stats': {
            'windows_total': stats.windows_total,
            'windows_after_type_filter': stats.windows_after_type_filter,
            'combinations_total': stats.combinations_total,
            'combinations_matched': stats.combinations_matched,
            'pair_lookups': stats.pair_lookups,
            'theta_d_bins_scanned': stats.theta_d_bins_scanned,
        },
        'top_paths': [
            {'path': list(map(int, path)), 'frame_count': int(count), 'frames': sorted(map(int, frames))}
            for (path, frames, count) in top_paths
        ],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    pipeline_elapsed_s = float(payload["pipeline_elapsed_s"])
    query_elapsed_s = float(payload["query_elapsed_s"])
    print(
        f'Done. pipeline={pipeline_elapsed_s:.3f}s query={query_elapsed_s:.3f}s '
        f'windows={stats.windows_after_type_filter}/{stats.windows_total} '
        f'combos={stats.combinations_total} matched={stats.combinations_matched}'
    )
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
