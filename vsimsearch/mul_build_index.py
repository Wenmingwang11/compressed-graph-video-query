from __future__ import annotations

import pickle
import time
from pathlib import Path

from vsimsearch.graph import compute_edges_from_nodes
from vsimsearch.io import read_nodes_from_mot_result_multi_class
from vsimsearch.indexing import build_index, build_node_index
from vsimsearch.untils import discretize_function_4

from subsets import create_node_dict
from cp_graph import create_cp_graph

REPO_ROOT = Path(__file__).resolve().parents[1]


def discretization_tag(theta_n_parts: int, theta_d_parts: int) -> str:
    return f"theta{theta_n_parts}_d{theta_d_parts}"


def artifact_paths(
    video_name: str,
    s: float,
    output_tag: str | None = None,
    base_dir: str | Path | None = None,
):
    base = Path(base_dir) if base_dir is not None else (REPO_ROOT / "storage" / "index")
    stem = f"{video_name}_{s}"
    if output_tag:
        stem = f"{stem}_{output_tag}"
    return (
        base / f"{stem}_index.pkl",
        base / f"{stem}_cp_graphs.pkl",
        base / f"{stem}_windowid_index.pkl",
        base / "build_cp_graph_time" / f"{stem}_time.txt",
    )


def slice_window_frame_objects(
    frame_objects,
    key_frame_indices,
    i: int,
):
    """
    Convert 1-based key-frame numbers into Python list slices.

    `find_key_frames()` writes frame numbers starting from 1, while
    `frame_objects` is a 0-based Python list. Each window should cover:
      [key_i, key_{i+1}) in frame-number space
    which becomes:
      [key_i - 1, key_{i+1} - 1) in list-slice space.
    """
    start = key_frame_indices[i] - 1
    if start < 0:
        raise ValueError(f"invalid key frame index: {key_frame_indices[i]}")
    if i + 1 < len(key_frame_indices):
        end = key_frame_indices[i + 1] - 1
        return frame_objects[start:end]
    return frame_objects[start:]


def main(
    video_name,
    j,
    theta_n_parts: int = 10,
    theta_d_parts: int = 8,
    output_tag: str | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    edge_build_mode: str = "normal",
):
    file_path = REPO_ROOT / "storage" / f"{video_name}.txt"
    nodes = read_nodes_from_mot_result_multi_class(str(file_path))

    subsection_file_path = REPO_ROOT / "storage" / "zhunbei" / f"{video_name}_{j}_subsection.pkl"
    with open(subsection_file_path, "rb") as f:
        key_frame_indices = pickle.load(f)
    frame_ids_file_path = REPO_ROOT / "storage" / f"{video_name}.txt.frame-ids.pkl"
    with open(frame_ids_file_path, "rb") as f:
        frame_objects = pickle.load(f)

    index = dict()
    cp_graphs = dict()
    windowid_index = dict()
    stime = time.time()
    width = 1920 if frame_width is None else int(frame_width)
    height = 1080 if frame_height is None else int(frame_height)
    edges = compute_edges_from_nodes(nodes, height, width, build_mode=edge_build_mode)
    discretize_function_4(edges, theta_n_parts=theta_n_parts, theta_d_parts=theta_d_parts)
    print(time.time() - stime)

    for i in range(len(key_frame_indices)):
        if i + 1 < len(key_frame_indices):
            edges_filtered = edges.loc[
                (edges["frame"] >= key_frame_indices[i]) & (edges["frame"] < key_frame_indices[i + 1])
            ]
            nodes_filtered = nodes.loc[
                (nodes["frame"] >= key_frame_indices[i]) & (nodes["frame"] < key_frame_indices[i + 1])
            ]
        else:
            edges_filtered = edges.loc[edges["frame"] >= key_frame_indices[i]]
            nodes_filtered = nodes.loc[nodes["frame"] >= key_frame_indices[i]]
        sect_frame_ids = slice_window_frame_objects(frame_objects, key_frame_indices, i)

        if not sect_frame_ids:
            continue

        graph_node_edges_index = build_node_index(edges_filtered)
        windowid_index[i + 1] = graph_node_edges_index

        node_dict, start_node = create_node_dict(sect_frame_ids)
        if not node_dict:
            continue

        graph = create_cp_graph(node_dict, start_node, graph_node_edges_index)
        cp_graphs[i + 1] = graph
        build_index(nodes_filtered, index, window_id=i + 1)

    etime = time.time()
    print(etime - stime)
    print(len(index))

    index_file_path, cp_graphs_file_path, windowid_index_file_path, time_file_path = artifact_paths(
        video_name,
        j,
        output_tag=output_tag,
    )
    time_file_path.parent.mkdir(parents=True, exist_ok=True)
    index_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(time_file_path, "w") as f:
        f.write(str(etime - stime))

    with open(index_file_path, "wb") as f:
        pickle.dump(index, f)

    with open(cp_graphs_file_path, "wb") as f:
        pickle.dump(cp_graphs, f)

    with open(windowid_index_file_path, "wb") as f:
        pickle.dump(windowid_index, f)

    return index_file_path, cp_graphs_file_path, windowid_index_file_path, time_file_path


if __name__ == "__main__":
    videos = ["drtrain"]
    bili = [0.9]
    for v in videos:
        for i in bili:
            main(v, i)
