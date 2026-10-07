import pickle
import itertools
import time

def get_from_multi_level_dict_with_tolerance(some_dict, keys, theta_tolerance=0.1, d_tolerance=0.1):
    """
    从多层嵌套字典中查询，并支持 theta 和 d 的容忍范围。
    """
    current_dict = some_dict
    for key in keys:
        if isinstance(key, tuple) and len(key) == 2:
            # Handle (theta, d) key
            theta, d = key
            sub_matches = {}
            for sub_key, value in current_dict.items():
                sub_theta, sub_d = sub_key
                if (theta is None or abs(sub_theta - theta) <= theta_tolerance) and (abs(sub_d-d) <= d_tolerance):
                    sub_matches = merge_dicts(sub_matches, value)
            if not sub_matches:
                return None
            current_dict = sub_matches
        else:
            current_dict = current_dict.get(key)
            if current_dict is None:
                return None
    return current_dict

def merge_dicts(dict1, dict2):
    """
    合并两个字典，处理 sid, eid -> frame_list 的合并。
    """
    if not dict1:
        return dict2
    if not dict2:
        return dict1

    result = dict1.copy()
    for key, value2 in dict2.items():
        if key in result:
            value1 = result[key]
            result[key] = list(set(value1) | set(value2))  # 合并 frame_list 去重
        else:
            result[key] = value2
    return result

def load_index(file_path):
    """
    从 pkl 文件中加载索引。
    """
    with open(file_path, 'rb') as f:
        index = pickle.load(f)
    return index

def print_query_results(results, output_file=None):
    """
    打印查询结果到控制台，并可选写入文件。
    """
    with open(output_file, 'w') if output_file else None as f:
        for obj_types, query_result in results.items():
            output_line = f"对象组合: {obj_types}\n"
            print(output_line, end='')
            if f:
                f.write(output_line)
            if query_result:
                for sid_eid, frame_list in query_result.items():
                    result_line = f"  (sid, eid): {sid_eid}, frame_list: {frame_list}\n"
                    print(result_line, end='')
                    if f:
                        f.write(result_line)
            else:
                no_result_line = "  无结果\n"
                print(no_result_line, end='')
                if f:
                    f.write(no_result_line)

# 主程序
if __name__ == "__main__":
    # 加载索引文件
    index_file = "./storage/baseline_index/bdd100kB-1-df0.pkl"  # 替换为你的文件路径
    index = load_index(index_file)
    # 打印索引中前十个键及其对应的值
    # for key, value in itertools.islice(index.items(), 10):
    #     print(f"Key: {key}, Value: {value}")
    print(index[1])
    print("索引加载成功！")

    # 查询参数
    query_objects_types = [[2, 4]]  # 替换为你的对象类型组合
    # query_objects_types = [[8, 3], [3, 3, 8], [3, 3, 8, 8]]  # 替换为你的对象类型组合
    query_spatial = (-0.4238413244435049, 0.05178606152034245)

    # 记录开始时间
    start_time = time.time()

    # 执行查询
    results = {}
    for obj_types in query_objects_types:
        for stype, etype in itertools.combinations(obj_types, 2):
            combined_keys = [stype, etype, query_spatial]  # 组合键
            print(combined_keys)
            query_result = get_from_multi_level_dict_with_tolerance(index, combined_keys)
            if query_result:
                if (stype, etype) in results:
                    results[(stype, etype)] = merge_dicts(results[(stype, etype)], query_result)
                else:
                    results[(stype, etype)] = query_result
    end_time = time.time()

    query_time = end_time - start_time
    # 打印查询结果到文件
    output_file = "./storage/query_results.txt"  # 查询结果保存文件
    print_query_results(results, output_file=output_file)
    print(f"查询结果已写入 {output_file}")

    # 将查询时间写入文件
    with open(output_file, 'a') as f:
        time_line = f"\n查询时间: {query_time:.6f} 秒\n"
        f.write(time_line)
        print(time_line)


# if __name__ == '__main__':
#     # 示例调用
#     videos = ['drtest', 'drtrain']
#     # videos = ['bdd100kA', 'bdd100kB']
#     query_objects_types = [[5, 10], [3, 5, 10], [3, 5, 8, 10]]  # 根据需要调整
#     query_spatial = (10, 5)
#     # similiarity = [0.6, 0.7, 0.8, 0.9]
#     similiarity = [0.6]
#     key = 10
#     for v in videos:
#         for s in similiarity:
#             total_time = batch_query(query_objects_types, query_spatial, v, key, s)
#             print(f"Total Time for {v}: {total_time}")
