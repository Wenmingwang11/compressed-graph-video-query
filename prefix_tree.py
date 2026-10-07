class TreeNode:
    def __init__(self):
        self.children = {}  # 存储子节点
        self.video_objects = set()  # 存储视频对象集合
        self.frames = set()  # 存储帧集合


def build_tree_from_results(results):
    root = TreeNode()  # 创建根节点

    for window_id, node_pairs in results.items():
        for node_pair, frames in node_pairs.items():
            current_node = root

            # 遍历节点组合，构建路径
            for node in node_pair:
                if node not in current_node.children:
                    current_node.children[node] = TreeNode()  # 创建子节点
                current_node = current_node.children[node]  # 移动到子节点

            # 在虚拟叶子节点中合并视频对象和帧
            current_node.video_objects.update(node_pair)  # 存储视频对象
            current_node.frames.update(frames)  # 合并视频帧集合

    return root

def is_consecutive(frames):
    sorted_frames = sorted(frames)
    return all(sorted_frames[i] + 1 == sorted_frames[i + 1] for i in range(len(sorted_frames) - 1))

# def collect_paths(node, current_path, path_frames, all_paths, threshold):
#     if node.video_objects:
#         # 检查帧集合的长度
#         if len(path_frames) > threshold:
#             # 检查帧的连续性
#             if is_consecutive(path_frames):
#                 all_paths.append((current_path, path_frames))
#
#             # 将帧集合存储到当前节点
#             node.frames.update(path_frames)
#
#     for child_id, child_node in node.children.items():
#         collect_paths(child_node, current_path + [child_id], path_frames.union(child_node.video_objects), all_paths, threshold)

# def collect_paths(node, current_path, path_frames, all_paths, threshold):
#     if node.video_objects:
#         # 检查帧集合的长度
#         if len(path_frames) > threshold:
#             # 检查帧的连续性并获取连续帧的数量
#             if is_consecutive(path_frames):
#                 continuous_count = len(path_frames)
#                 all_paths.append((current_path, continuous_count))
#
#     for child_id, child_node in node.children.items():
#         collect_paths(child_node, current_path + [child_id], path_frames.union(child_node.video_objects), all_paths, threshold)

def collect_paths(node, current_path, all_paths, threshold):
    # 仅在帧集合超过阈值时收集路径
    if len(node.frames) > threshold:
        all_paths.append((current_path, node.frames.copy()))  # 记录路径和帧集合

    for child_id, child_node in node.children.items():
        collect_paths(child_node, current_path + [child_id], all_paths, threshold)

# def get_top_five_consecutive_paths(root, threshold):
#     all_paths = []
#     collect_paths(root, [], set(), all_paths, threshold)
#
#     # 计算每条路径的连续帧数量
#     path_consecutive_counts = [(path, len(frames)) for path, frames in all_paths]
#
#     # 按照连续帧数量降序排序
#     top_five = sorted(path_consecutive_counts, key=lambda x: x[1], reverse=True)[:5]
#
#     return top_five

# def get_top_five_longest_consecutive_paths(root, threshold):
#     all_paths = []
#     collect_paths(root, [], set(), all_paths, threshold)
#
#     # 根据连续帧数量进行排序
#     top_five = sorted(all_paths, key=lambda x: x[1], reverse=True)[:5]
#
#     return top_five

def get_top_five_paths_with_threshold(root, threshold, key):
    all_paths = []
    collect_paths(root, [], all_paths, threshold)

    # 计算每条路径的帧数量
    path_frame_counts = [(path, frames, len(frames)) for path, frames in all_paths]

    # 按照帧数量降序排序
    top_five = sorted(path_frame_counts, key=lambda x: x[2], reverse=True)[:key]

    return top_five
