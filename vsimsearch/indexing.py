from vsimsearch.untils import compute_if_absent, get_from_multi_level_dict, init_bitset
from collections import defaultdict

# def build_index(nodes, index, window_id):
#     '''
#     索引格式:
#       node_type -> node_id -> window_id
#     '''
#     id_type_dict = {int(node['id']): int(node['class']) for _, node in nodes.iterrows()}
#
#     for node_id, node_type in id_type_dict.items():
#         type_index = compute_if_absent(index, node_type, dict)
#         id_index = compute_if_absent(type_index, node_id, set)
#         id_index.add(window_id)
def build_index(nodes, index, window_id):
    '''
    索引格式:
      window_id -> node_type -> node_id
    '''
    id_type_dict = {int(node['id']): int(node['class']) for _, node in nodes.iterrows()}

    for node_id, node_type in id_type_dict.items():
        window_index = compute_if_absent(index, window_id, dict)
        type_index = compute_if_absent(window_index, node_type, set)
        type_index.add(node_id)

def build_node_index(edges):
    '''
    Index format:
      sid, eid -> theta, d -> frame_list
    '''
    # 按帧分组节点和边
    # grouped_nodes = nodes.groupby('frame')
    grouped_edges = edges.groupby('frame')

    # 初始化索引字典
    index = dict()

    # 遍历每一帧的边
    for fid in grouped_edges.groups.keys():
        # subtable_nodes = grouped_nodes.get_group(fid)
        subtable_edges = grouped_edges.get_group(fid)


        # 遍历当前帧的每条边
        for _, row in subtable_edges.iterrows():
            sid, eid, theta, d = int(row['sid']), int(row['eid']), row['theta'], row['d_ratio']

            # 在索引中创建相应的嵌套结构
            sid_eid_index = compute_if_absent(index, (sid, eid), dict)
            theta_d_index = compute_if_absent(sid_eid_index, (theta, d), set)

            # 将当前帧ID添加到帧列表中
            theta_d_index.add(fid)

    return index
