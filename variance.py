import numpy as np
import pickle

def calculate_variance(data):
    # 将数据转换为元组
    data_as_tuples = [tuple(obj) for obj in data]

    # 将元组转换为整体值（例如字符串表示）
    data_as_strings = [str(obj) for obj in data_as_tuples]

    # 计算字符串的哈希值作为整体值
    data_as_hashes = [hash(obj) for obj in data_as_strings]

    # 计算方差
    variance = np.var(data_as_hashes)

    return variance
with open('storage/drtest.txt.frame-ids.pkl', 'rb') as f:
    lists = pickle.load(f)
frame_objects = lists[:10]
# # 输入数据
frame_objects = [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [2, 3, 4],
    [2, 3, 4],
    [2, 3, 4],
    [3, 4, 5],
    [3, 4, 5],
    [3, 4, 5]
]
# frame_objects = [
#     [1, 2, 3],
#     [2, 3, 4],
#     [3, 4, 5],
#     [4, 5, 6],
#     [5, 6, 7],
#     [6, 7, 8],
#     [7, 8, 9],
#     [8, 9, 10],
#     [9, 10, 11]
# ]


# 比例设定
variance_multiplier_threshold = 2

# 初始化
current_segment = [frame_objects[0]]
segments = []
all_variances = []
change_indices = []

# 递增计算方差
for i in range(1, len(frame_objects)):
    next_element = frame_objects[i]
    new_segment = current_segment + [next_element]

    # 计算当前段的方差
    current_variance = calculate_variance(new_segment)

    # 计算之前所有方差的平均值
    if all_variances:
        mean_variance = np.mean(all_variances)
    else:
        mean_variance = 0  # 如果还没有方差，设定为0（可以根据需求调整）

    # 比较方差变化
    if current_variance > variance_multiplier_threshold * mean_variance:
        # 如果当前方差比平均方差高出一定比例，保存当前段，并从新段开始
        segments.append(current_segment)
        current_segment = [next_element]
        all_variances = [current_variance]  # 重新开始计算新的段的方差列表
        change_indices.append(i)
    else:
        # 否则，继续在当前段上添加新元素
        current_segment = new_segment
        all_variances.append(current_variance)  # 添加当前方差到方差列表

# 添加最后一段
if current_segment:
    segments.append(current_segment)

# # 打印结果
# for i, segment in enumerate(segments):
#     print(f"段 {i + 1}:")
#     for obj in segment:
#         print(obj)
#     print("\n")

# 输出发生显著变化的编号
print("发生显著变化的编号:", change_indices)
