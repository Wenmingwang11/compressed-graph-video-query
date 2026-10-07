from Tree_Node import treeNode
import time

def create_tree(data_set, min_support=1):
    """
    创建FP树
    :param data_set: 数据集
    :param min_support: 最小支持度
    :return:
    """
    freq_items = {}  # 频繁项集
    for trans in data_set:  # 第一次遍历数据集
        for item in trans:
            if item in freq_items:
                freq_items[item] += 1
            else:
                freq_items[item] = 1

    header_table = {k: v for k, v in freq_items.items() if v >= min_support}  # 创建头指针表

    # 无频繁项集
    if len(header_table) == 0:
        return None, None

    for k in header_table:
        header_table[k] = [header_table[k], None]  # 添加头指针表指向树中的数据

    # 创建树过程
    ret_tree = treeNode('Null Set', 1, None)  # 根节点

    # 第二次遍历数据集
    for trans in data_set:  # 注意这里遍历的是 data_set 中的每个交易
        local_data = {}
        for item in trans:
            if item in header_table:
                local_data[item] = header_table[item][0]

        if len(local_data) > 0:
            ordered_items = [v[0] for v in sorted(local_data.items(), key=lambda kv: (-kv[1], kv[0]))]
            update_tree(ordered_items, ret_tree, header_table, 1)  # 这里使用 1 作为 count，因为 data_set 中的每个交易都是频繁的

    return ret_tree, header_table




def update_tree(items, in_tree, header_table, count):
    '''
    :param items: 元素项
    :param in_tree: 检查当前节点
    :param header_table:
    :param count:
    :return:
    '''
    if items[0] in in_tree.children:  # check if ordered_items[0] in ret_tree.children
        in_tree.children[items[0]].increase(count)  # incrament count
    else:  # add items[0] to in_tree.children
        in_tree.children[items[0]] = treeNode(items[0], count, in_tree)
        if header_table[items[0]][1] is None:  # update header table
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            update_header(header_table[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:  # call update_tree() with remaining ordered items
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)


def update_header(node_test, target_node):
    '''
    :param node_test:
    :param target_node:
    :return:
    '''
    while node_test.node_link is not None:  # Do not use recursion to traverse a linked list!
        node_test = node_test.node_link
    node_test.node_link = target_node


def ascend_tree(leaf_node, pre_fix_path):
    '''
    遍历父节点，找到路径
    :param leaf_node:
    :param pre_fix_path:
    :return:
    '''
    if leaf_node.parent is not None:
        pre_fix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, pre_fix_path)


def find_pre_fix_path(base_pat, tree_node):
    '''
    创建前缀路径
    :param base_pat: 频繁项
    :param treeNode: FP树中对应的第一个节点
    :return:
    '''
    # 条件模式基
    cond_pats = {}
    while tree_node is not None:
        pre_fix_path = []
        ascend_tree(tree_node, pre_fix_path)
        if len(pre_fix_path) > 1:
            cond_pats[frozenset(pre_fix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    # print(cond_pats)
    return cond_pats


def mine_tree(in_tree, header_table, min_support, pre_fix, freq_items):
    '''
    挖掘频繁项集
    :param in_tree:
    :param header_table:
    :param min_support:
    :param pre_fix:
    :param freq_items:
    :return:
    '''

    # 从小到大排列table中的元素，为遍历寻找频繁集合使用
    bigL = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]  # (sort header table by frequency)
    for base_pat in bigL:  # start from bottom of header table
        new_freq_set = pre_fix.copy()
        new_freq_set.add(base_pat)
        if len(new_freq_set) > 0:
            freq_items[frozenset(new_freq_set)] = header_table[base_pat][0]
        cond_patt_bases = find_pre_fix_path(base_pat, header_table[base_pat][1])
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_support)

        if my_head is not None:
            mine_tree(my_cond_tree, my_head, min_support, new_freq_set, freq_items)



def fp_growth(data_set, min_support=1):
    my_fp_tree, my_header_tab = create_tree(data_set, min_support)
    # bigL = [v[0] for v in sorted(my_header_tab.items(), key=lambda p: p[1][0], reverse=True)]
    conditional_pattern_bases = {}
    for item in my_header_tab:
        conditional_pattern_bases[item] = find_pre_fix_path(item, my_header_tab[item][1])
    # print(conditional_pattern_bases)
    frequent_itemsets = []
    for item, pattern_base in conditional_pattern_bases.items():
        if pattern_base:
            current_frequent_itemsets = []
            for pattern, count in pattern_base.items():
                new_pattern = pattern | {item}
                current_frequent_itemsets.append((new_pattern, count))
            # print(current_frequent_itemsets)
            for itemset, count in current_frequent_itemsets:
                updated_count = count
                for subset, subset_count in current_frequent_itemsets:
                    if subset != itemset and itemset.issubset(subset):  # 判断是否为子集
                        updated_count += subset_count  # 加上包含该子集的项集的计数
                frequent_itemsets.append((itemset, updated_count))


    frequent_1_itemsets = {frozenset([item]): my_header_tab[item][0] for item in my_header_tab}
    # print(frequent_1_itemsets)
    for one_itemset, one_count in frequent_1_itemsets.items():
        is_valid = True
        for itemset, count in frequent_itemsets:
            if one_itemset.issubset(itemset) and one_count == count:
                is_valid = False
                break
        if is_valid:
            frequent_itemsets.append((one_itemset, one_count))
    itemset_dict = {}
    for itemset, count in frequent_itemsets:
        itemset_dict[tuple(itemset)] = count
    return itemset_dict
    # my_fp_tree.disp()


if __name__ == '__main__':
    # 示例数据集
    data_set = [[1, 2], [2, 3], [1,2,3], [1, 2, 3, 4], [1,2,3,4,5]]
    # data_set = [[811, 882, 847, 766, 871, 866, 857, 606, 655, 883, 842, 439, 733, 839, 860, 848, 888, 891, 864, 854, 889], [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 864, 871, 882, 883, 888, 889, 891, 895], [439, 606, 655, 733, 766, 785, 811, 839, 842, 847, 854, 857, 860, 864, 871, 875, 882, 883, 889, 891, 895, 896], [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 871, 875, 881, 882, 883, 889, 891, 895, 896], [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 875, 882, 883, 889, 891, 895, 896], [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 871, 875, 881, 882, 883, 889, 891, 894, 895, 896, 897, 898], [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 871, 875, 881, 882, 883, 889, 894, 895, 896, 899], [439, 606, 655, 733, 766, 811, 839, 842, 847, 854, 860, 871, 875, 881, 882, 883, 889, 891, 894, 895, 896, 898, 899, 900, 901], [439, 606, 655, 733, 766, 811, 839, 842, 847, 854, 860, 866, 871, 881, 882, 883, 889, 894, 895, 896, 899, 900, 902, 903], [439, 606, 655, 733, 766, 811, 839, 842, 847, 854, 860, 866, 871, 882, 883, 889, 894, 895, 896, 898, 899, 900, 901, 902, 904, 905], [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 860, 866, 871, 875, 881, 882, 883, 891, 894, 895, 896, 898, 900, 902, 904, 906, 907], [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 860, 866, 871, 875, 881, 882, 883, 889, 891, 894, 895, 896, 898, 900, 902, 904, 906], [439, 606, 655, 733, 766, 811, 839, 847, 857, 860, 866, 871, 881, 882, 883, 889, 891, 894, 895, 896, 898, 900, 902, 906], [439, 606, 655, 733, 811, 839, 847, 854, 857, 860, 871, 881, 882, 883, 889, 891, 894, 895, 896, 898, 900, 902], [439, 606, 655, 733, 811, 839, 847, 857, 860, 871, 881, 882, 889, 894, 895, 896, 898, 900, 902, 905], [439, 606, 655, 733, 811, 839, 847, 848, 860, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 902], [606, 655, 733, 766, 811, 839, 847, 848, 860, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 902], [606, 655, 733, 766, 811, 839, 847, 848, 860, 866, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900], [606, 655, 733, 811, 839, 847, 848, 857, 860, 866, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 908], [606, 655, 733, 766, 811, 839, 847, 848, 857, 860, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 908, 909], [439, 606, 655, 733, 766, 811, 839, 847, 857, 860, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910], [606, 655, 733, 766, 811, 839, 847, 860, 871, 875, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 911, 912], [606, 655, 733, 766, 811, 839, 847, 860, 871, 875, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 912], [606, 655, 733, 766, 811, 839, 847, 860, 871, 875, 881, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913], [606, 655, 733, 766, 811, 839, 847, 860, 871, 875, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913], [606, 655, 733, 766, 811, 839, 847, 848, 860, 871, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913, 914, 915], [439, 606, 655, 733, 766, 811, 839, 848, 860, 871, 875, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913, 915, 916], [606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913, 915, 917], [606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 912, 913, 915, 918], [606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913, 915, 919, 920], [439, 606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 913, 915, 921], [439, 606, 733, 766, 811, 848, 860, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 913, 915, 920], [439, 606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 882, 889, 894, 895, 896, 900, 908, 909, 913, 915, 919, 922, 923], [439, 733, 766, 811, 848, 860, 875, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 913, 915, 919, 920, 923, 924, 925, 926], [439, 655, 733, 766, 811, 848, 860, 875, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 913, 915, 923, 924, 925, 926, 927, 928], [439, 733, 766, 811, 848, 860, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 915, 919, 923, 924, 925, 927, 929], [439, 655, 733, 811, 848, 860, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 915, 919, 923, 924, 925, 928, 930, 931], [439, 655, 733, 811, 848, 875, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 913, 915, 919, 923, 924, 925], [439, 655, 848, 875, 881, 882, 889, 891, 894, 895, 896, 900, 909, 910, 913, 915, 919, 923, 924, 925, 926, 928, 929, 932]]
    # data_set = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 10], [3, 4, 5, 6, 10, 11], [4, 5, 6, 10, 11, 12],
    #             [5, 6, 10, 11, 12, 13], [6, 10, 11, 12, 13, 14], [10, 11, 12, 13, 14, 15]]
    print(len(data_set))

    # 转换数据集格式为适合FP-growth算法的形式
    # data_set = [set(trans) for trans in dataset]

    # 最小支持度
    min_support = 1
    s_time = time.time()
    # 获取频繁项集与条件模式基
    freq_items = fp_growth(data_set, min_support)

    print(freq_items)
    print(len(freq_items))
    print(time.time() - s_time)
    # 打印结果
    # for itemset, count in frequent_itemsets_with_conditional_bases:
    #     print(f"Frequent Itemset: {itemset}, Count: {count}")
    # print(freq_items)
