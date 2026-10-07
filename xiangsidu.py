def calculate_difference(list1, list2):
    return sum(1 for x, y in zip(list1, list2) if x != y)

def find_significant_change_index(frame_objects, threshold):
    change_indices = []
    last_change_index = 0

    for i in range(1, len(frame_objects)):
        total_change = calculate_difference(frame_objects[i], frame_objects[last_change_index])
        if total_change > threshold:
            change_indices.append(i)
            last_change_index = i  # 更新变化较大的列表索引

    return change_indices

# frame_objects = [
#     [1, 2, 3],
#     [1, 2, 3],
#     [1, 2, 3],
#     [2, 3, 4],
#     [2, 3, 4],
#     [2, 3, 4],
#     [3, 4, 5],
#     [3, 4, 5],
#     [3, 4, 5]
# ]
frame_objects = [
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9],
    [8, 9, 10],
    [9, 10, 11]
]

threshold = 1  # 设置一个阈值，定义“显著变化”的程度
change_indices = find_significant_change_index(frame_objects, threshold)
print(f"显著变化发生在以下索引处: {change_indices}")
