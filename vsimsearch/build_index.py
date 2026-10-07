import sys
import os, time
from vsimsearch.graph import compute_edges_from_nodes
from vsimsearch.io import read_nodes_from_mot_result_multi_class
from vsimsearch.indexing import build_index, build_node_index
from vsimsearch.untils import discretize_function_4
import pickle
import tracemalloc
import gc
# 获取当前文件的上级目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将上级目录添加到sys.path中
sys.path.append(parent_dir)
from subsets import create_node_dict
from cp_graph import create_cp_graph
if __name__ == '__main__':
    file_path = '../storage/bdd100kA.txt'
    nodes = read_nodes_from_mot_result_multi_class(file_path)
    with open('../storage/bdd100kA_subsection.pkl', 'rb') as f:
        key_frame_indices = pickle.load(f)
    # print(key_frame_indices)
    # print(len(key_frame_indices))
    with open('../storage/bdd100kA.txt.frame-ids.pkl', 'rb') as f:
        frame_objects = pickle.load(f)
    index = dict()
    cp_graphs = dict()
    stime = time.time()
    tracemalloc.start()
    for i in range(len(key_frame_indices)):
        if i + 1 < len(key_frame_indices):
            # print(frame_objects[key_frame_indices[i]:key_frame_indices[i + 1]])
            # 使用 .loc 和 & 来组合条件
            nodes_filtered = nodes.loc[
                (nodes['frame'] >= key_frame_indices[i]) & (nodes['frame'] < key_frame_indices[i + 1])]
            sect_frame_ids = frame_objects[key_frame_indices[i]:key_frame_indices[i + 1]]
        else:
            nodes_filtered = nodes.loc[nodes['frame'] >= key_frame_indices[i]]
            sect_frame_ids = frame_objects[key_frame_indices[i]:]
        edges = compute_edges_from_nodes(nodes_filtered, 1080, 1920)
        discretize_function_4(edges)
        graph_node_edges_index = build_node_index(nodes_filtered, edges)
        print(i)
        node_dict, start_node = create_node_dict(sect_frame_ids)
        if not node_dict:
            continue
        print("sect_frame_ids", sect_frame_ids)
        G = create_cp_graph(node_dict, start_node, graph_node_edges_index)
        cp_graphs[i + 1] = G
        build_index(nodes_filtered, index, window_id=i+1)
    etime = time.time()
    # print(index)
    # print(len(index))
    # print(cp_graphs)
    # print(len(cp_graphs))
