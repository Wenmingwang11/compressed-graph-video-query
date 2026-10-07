import sys
import os
import time
import pickle
from vsimsearch.graph import compute_edges_from_nodes
from vsimsearch.io import read_nodes_from_mot_result_multi_class
from vsimsearch.indexing import build_index, build_node_index
from vsimsearch.untils import discretize_function_4

# 获取当前文件的上级目录路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将上级目录添加到sys.path中
sys.path.append(parent_dir)
from subsets import create_node_dict
from cp_graph import create_cp_graph

def filted_nodes(nodes):
    nodes = nodes.loc[(nodes['frame'] >= 10) & (nodes['frame'] < 20)]
    return nodes

def main():
    file_path = '../storage/drtrain.txt'
    nodes = read_nodes_from_mot_result_multi_class(file_path)

    with open('../storage/zhunbei/drtrain_0.6_subsection.pkl', 'rb') as f:
        key_frame_indices = pickle.load(f)

    with open('../storage/drtrain.txt.frame-ids.pkl', 'rb') as f:
        frame_objects = pickle.load(f)

    index = dict()
    cp_graphs = dict()
    windowid_index = dict()
    stime = time.time()
    edges = compute_edges_from_nodes(nodes, 1080, 1920)
    discretize_function_4(edges)
    print(time.time()-stime)

    for i in range(len(key_frame_indices)):
        if i + 1 < len(key_frame_indices):
            edges_filtered = edges.loc[
                (edges['frame'] >= key_frame_indices[i]) & (edges['frame'] < key_frame_indices[i + 1])]
            nodes_filtered = nodes.loc[
                (nodes['frame'] >= key_frame_indices[i]) & (nodes['frame'] < key_frame_indices[i + 1])]
            sect_frame_ids = frame_objects[key_frame_indices[i]:key_frame_indices[i + 1]]
        else:
            edges_filtered = edges.loc[edges['frame'] >= key_frame_indices[i]]
            nodes_filtered = nodes.loc[nodes['frame'] >= key_frame_indices[i]]
            sect_frame_ids = frame_objects[key_frame_indices[i]:]


        graph_node_edges_index = build_node_index(edges_filtered)
        windowid_index[i+1] = graph_node_edges_index
        print(i)

        # Uncomment these lines if needed
        node_dict, start_node = create_node_dict(sect_frame_ids)
        if not node_dict:
            continue
        G = create_cp_graph(node_dict, start_node, graph_node_edges_index)
        cp_graphs[i + 1] = G
        build_index(nodes_filtered, index, window_id=i+1)
        # if i == 100:
        #     break

    etime = time.time()
    print(etime - stime)
    print(len(index))
    with open('../storage/index/drtrain_index.pkl', 'wb') as f:
        pickle.dump(index, f)

    # 将 cp_graphs 存储到 pkl 文件
    with open('../storage/index/drtrain_cp_graphs.pkl', 'wb') as f:
        pickle.dump(cp_graphs, f)

    with open('../storage/index/drtrain_windowid_index.pkl', 'wb') as f:
        pickle.dump(windowid_index, f)

if __name__ == '__main__':
    main()
