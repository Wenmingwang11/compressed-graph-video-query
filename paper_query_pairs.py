import argparse
import pickle
import time

from vsimsearch.index_files import resolve_index_paths
from vsimsearch.paper_querying_pairs import (
    parse_pair_constraints,
    run_paper_query_with_pair_constraints,
)


def run_single_query_with_pair_constraints(
    video_name: str,
    s: float | None,
    query_types: list[int],
    pair_constraints_text: str,
    topk: int,
    frame_threshold: int,
    max_bindings_per_window: int = 0,
):
    index_path, _, cp_graphs_path = resolve_index_paths(video_name, s)
    if cp_graphs_path is None:
        raise FileNotFoundError(f"Compressed graph index not found for video={video_name}, s={s}")

    with open(index_path, "rb") as f:
        index = pickle.load(f)
    with open(cp_graphs_path, "rb") as f:
        cp_graphs = pickle.load(f)

    pair_constraints = parse_pair_constraints(pair_constraints_text, len(query_types))

    start = time.time()
    stats, top_paths = run_paper_query_with_pair_constraints(
        index=index,
        cp_graphs=cp_graphs,
        query_object_types=query_types,
        pair_constraints=pair_constraints,
        topk=topk,
        frame_threshold=frame_threshold,
        max_bindings_per_window=max_bindings_per_window,
    )
    elapsed = time.time() - start
    return elapsed, stats, top_paths, pair_constraints


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run one structured query on compressed graphs with pair-specific role constraints"
    )
    parser.add_argument("--video", default="drtest")
    parser.add_argument("--s", type=float, default=None, help="window similarity threshold S used at indexing time")
    parser.add_argument("--types", type=str, default="0,1", help="comma-separated object types")
    parser.add_argument(
        "--pair_constraints",
        type=str,
        required=True,
        help="role-pair constraints like 0-1:1.0,5;0-2:0.5,2;1-2:2.0,8",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--frame_threshold", type=int, default=2)
    parser.add_argument("--max_bindings_per_window", type=int, default=0, help="0 means no cap")
    args = parser.parse_args()

    query_types = [int(x) for x in args.types.split(",") if x.strip() != ""]

    elapsed, stats, top_paths, pair_constraints = run_single_query_with_pair_constraints(
        video_name=args.video,
        s=args.s,
        query_types=query_types,
        pair_constraints_text=args.pair_constraints,
        topk=args.topk,
        frame_threshold=args.frame_threshold,
        max_bindings_per_window=args.max_bindings_per_window,
    )

    print(f"pair_constraints={pair_constraints}")
    print(
        f"elapsed={elapsed:.4f}s windows={stats.windows_after_type_filter}/{stats.windows_total} "
        f"candidate_vertices={stats.candidate_vertices_total} bindings={stats.bindings_total} "
        f"matched={stats.bindings_matched} pair_lookups={stats.pair_lookups} "
        f"bins_scanned={stats.theta_d_bins_scanned}"
    )
    for path, frames, count in top_paths:
        print("Path:", "->".join(map(str, path)), "Frame Count:", count, "Frames:", frames)
