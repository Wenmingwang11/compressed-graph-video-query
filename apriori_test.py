from itertools import combinations
import pickle
from time import time


def has_infrequent_subset(candidate: set, previous_L: list) -> bool:
    """
    A function to prone some of candidates

    Parameters
    ----------
    candidate -- a set to check whether all of its subsets are frequent or not.
        if any subset is not frequent, the function will returns True,
        otherwise returns False.
    previous_L -- a list of tuples to check candidate's subsets against it.
        an instance of previous_L could be found in 'Examples' part.

    Returns
    -------
    a boolean value. True means there are some subsets in the candidate that
    are not frequent with respect to previous_L and this value should no be
    included in the final Ck result. False means all subsets are frequent and
    we shall include this candidate in our Ck result.

    Examples
    --------
    # >>> previous_L = [(1,2,4),(2,3,6),(2,3,8),(2,6,7),(2,6,8),(3,4,5),(3,6,8)]
    # >>> has_infrequent_subset((2,3,6,8), previous_L)
    False
    # >>> has_infrequent_subset((2,3,6,7), previous_L)
    True
    """
    subsets = combinations(candidate, len(candidate)-1)
    for subset in subsets:  # subset is a tuple
        if subset not in previous_L:
            return True
    return False


def apriori_gen(previous_L: dict) -> dict:
    """
    A function generate candidates with respect to Lk-1 (previous_L). tries
    prone the results with the help of has_infrequent_subset(). for every new
    candidate found, if all of its subsets with the length of k-1 are not
    frequent in Lk-1 (previous_L), it will not be added to the result.
    """
    Ck = {}
    for item_1, TIDs1 in previous_L.items():
        for item_2, TIDs2 in previous_L.items():
            if item_1[:-1] == item_2[:-1] and item_1[-1] < item_2[-1]:
                new_item = tuple([*item_1, item_2[-1]])
                # if has_infrequent_subset(new_item, previous_L):
                #     continue
                new_TIDs = TIDs1.intersection(TIDs2)
                if not new_TIDs:
                    continue
                Ck[new_item] = new_TIDs
    return Ck


def generate_L1(dataset: list, min_support_count: int) -> dict:
    """
    Generates L1 itemset from given dataset with respect to min_support_count

    Parameters
    ----------
    dataset -- a list of lists. each inner list represent a transaction which
        its content are items bought in that transacton. the outer list is the
        dataset which contain all transactions.
    min_support_count -- an integer which is used to check whether one item is
        frequent or not.

    Returns
    -------
    a dictionary with keys representing L1 frequent items fount and values
    representing what transactions that item appeared in. the values are sets.
    the values will be useful later as this paper demonstrates:
    https://arxiv.org/pdf/1403.3948.pdf

    Examples
    --------
    >>> generate_L1([[1,2,3], [3,4,1], [3,4,5]], 3)
    {(3,): {0, 1, 2}}
    >>> generate_L1([[1,2,3], [3,4,1], [3,4,5]], 2)
    {(1,): {0, 1}, (3,): {0, 1, 2}, (4,): {1, 2}}
    """
    L1 = {}
    for TID, transaction in enumerate(dataset):
        for item in transaction:
            if (item,) not in L1:
                L1[(item,)] = set()
            L1[(item,)].add(TID)
    print(len(L1))
    return {item: TIDs for item, TIDs in L1.items()
            if len(TIDs) >= min_support_count}


# def gen_Lk(Ck: dict, dataset: list, min_support_count: int) -> dict:
#     st = time()
#     Lk = {}
#     for candidate, TIDs in Ck.items():
#         if len(TIDs) < min_support_count:
#             continue
#         Lk[candidate] = TIDs
#     return Lk


def gen_Lk(Ck: dict, dataset: list, min_support_count: int) -> dict:
    Lk = Ck
    print(len(Lk))
    return Lk


def apriori(min_support_count: int, dataset: list):
    st = time()
    L1 = generate_L1(dataset, min_support_count)
    L = {1: L1}
    for k in range(2, 1000):
        if len(L[k-1]) < 2:
            break
        Ck = apriori_gen(L[k-1])
        L[k] = gen_Lk(Ck, dataset, min_support_count)
    return L


if __name__ == '__main__':
    # dataset = [[1, 2, 3, 4, 5],
    #            [1, 2, 3, 4, 5, 6],
    #            [1, 2, 3, 4, 5, 6, 7],
    #            [1, 2, 3, 4, 8, 6, 7],
    #            [1, 2, 3, 4, 5, 6, 9]]
    # dataset = [[0, 5, 3, 27, 16, 19, 21, 23, 24],
    #            [0, 5, 3, 16, 19, 21, 23, 24],
    #            [5, 3, 0, 24, 28, 16, 19, 23],
    #            [0, 5, 3, 24, 28, 29, 16, 19, 21, 23],
    #            [0, 5, 3, 24, 30, 16, 19, 23],
    #            [5, 30, 0, 24, 3, 28, 16, 19, 21, 23],
    #            [24, 5, 30, 3, 0, 16, 19, 21, 23],
    #            [24, 30, 5, 3, 31, 16, 19, 32, 23],
    #            [24, 31, 5, 3, 30, 33, 34, 16, 19, 21, 23],
    #            [31, 24, 33, 3, 5, 30, 16, 19, 21, 23],
    #            [31, 24, 3, 5, 33, 35, 36, 37, 16, 19, 21, 23, 38],
    #            [24, 37, 31, 36, 39, 40, 41, 35, 16, 19, 21, 32, 23],
    #            [37, 31, 24, 39, 36, 40, 16, 19, 21, 42, 23],
    #            [37, 24, 31, 36, 43, 40, 44, 16, 21, 19, 45, 23],
    #            [37, 36, 43, 24, 31, 40, 16, 19, 21, 42, 23],
    #            [37, 31, 43, 24, 36, 40, 44, 42, 16, 19, 21],
    #            [37, 36, 31, 24, 44, 46, 40, 5, 42, 16, 21, 41],
    #            [37, 46, 36, 31, 24, 40, 42, 16, 21, 41, 23],
    #            [37, 36, 46, 31, 40, 42, 16, 21, 41],
    #            [37, 36, 46, 31, 40, 24, 42, 21],
    #            [37, 31, 24, 36, 46, 40, 42, 21],
    #            [37, 31, 24, 46, 36, 40, 47, 42, 21],
    #            [37, 31, 24, 36, 47, 40, 48, 49, 42, 21, 16]]
    # dataset = [[1122, 1071, 965, 1121, 1109, 884, 1124, 1130, 1093, 1016, 1120, 958, 1119, 1096, 1133, 1021, 1117, 900, 439, 951, 980, 724, 1014, 1004, 713, 790, 848, 1131, 859, 1036, 1039, 788, 750, 719, 944, 1101, 861, 1132, 1134, 1099, 1127, 1123, 566, 1026, 688, 1095, 1128, 1135, 806], [1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165]]
    dataset = [
        [811, 882, 847, 766, 871, 866, 857, 606, 655, 883, 842, 439, 733, 839, 860, 848, 888, 891, 864, 854, 889],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 864, 871, 882, 883, 888, 889, 891, 895],
        [439, 606, 655, 733, 766, 785, 811, 839, 842, 847, 854, 857, 860, 864, 871, 875, 882, 883, 889, 891, 895, 896],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 871, 875, 881, 882, 883, 889, 891, 895, 896],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 875, 882, 883, 889, 891, 895, 896],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 871, 875, 881, 882, 883, 889, 891, 894, 895,
         896, 897, 898],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 857, 860, 871, 875, 881, 882, 883, 889, 894, 895, 896,
         899],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 854, 860, 871, 875, 881, 882, 883, 889, 891, 894, 895, 896, 898,
         899, 900, 901],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 854, 860, 866, 871, 881, 882, 883, 889, 894, 895, 896, 899, 900,
         902, 903],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 854, 860, 866, 871, 882, 883, 889, 894, 895, 896, 898, 899, 900,
         901, 902, 904, 905],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 860, 866, 871, 875, 881, 882, 883, 891, 894, 895, 896,
         898, 900, 902, 904, 906, 907],
        [439, 606, 655, 733, 766, 811, 839, 842, 847, 848, 854, 860, 866, 871, 875, 881, 882, 883, 889, 891, 894, 895,
         896, 898, 900, 902, 904, 906],
        [439, 606, 655, 733, 766, 811, 839, 847, 857, 860, 866, 871, 881, 882, 883, 889, 891, 894, 895, 896, 898, 900,
         902, 906],
        [439, 606, 655, 733, 811, 839, 847, 854, 857, 860, 871, 881, 882, 883, 889, 891, 894, 895, 896, 898, 900, 902],
        [439, 606, 655, 733, 811, 839, 847, 857, 860, 871, 881, 882, 889, 894, 895, 896, 898, 900, 902, 905],
        [439, 606, 655, 733, 811, 839, 847, 848, 860, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 902],
        [606, 655, 733, 766, 811, 839, 847, 848, 860, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 902],
        [606, 655, 733, 766, 811, 839, 847, 848, 860, 866, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900],
        [606, 655, 733, 811, 839, 847, 848, 857, 860, 866, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 908],
        [606, 655, 733, 766, 811, 839, 847, 848, 857, 860, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 908, 909],
        [439, 606, 655, 733, 766, 811, 839, 847, 857, 860, 871, 881, 882, 889, 891, 894, 895, 896, 898, 900, 908, 909,
         910],
        [606, 655, 733, 766, 811, 839, 847, 860, 871, 875, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 911, 912],
        [606, 655, 733, 766, 811, 839, 847, 860, 871, 875, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 912],
        [606, 655, 733, 766, 811, 839, 847, 860, 871, 875, 881, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913],
        [606, 655, 733, 766, 811, 839, 847, 860, 871, 875, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913],
        [606, 655, 733, 766, 811, 839, 847, 848, 860, 871, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913, 914,
         915],
        [439, 606, 655, 733, 766, 811, 839, 848, 860, 871, 875, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913,
         915, 916],
        [606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913, 915,
         917],
        [606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 912, 913,
         915, 918],
        [606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 889, 891, 894, 895, 896, 898, 900, 908, 909, 910, 913, 915,
         919, 920],
        [439, 606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 913,
         915, 921],
        [439, 606, 733, 766, 811, 848, 860, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 913, 915, 920],
        [439, 606, 655, 733, 766, 811, 848, 860, 871, 875, 881, 882, 889, 894, 895, 896, 900, 908, 909, 913, 915, 919,
         922, 923],
        [439, 733, 766, 811, 848, 860, 875, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 913, 915, 919, 920, 923,
         924, 925, 926],
        [439, 655, 733, 766, 811, 848, 860, 875, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 913, 915, 923, 924,
         925, 926, 927, 928],
        [439, 733, 766, 811, 848, 860, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 915, 919, 923, 924, 925,
         927, 929],
        [439, 655, 733, 811, 848, 860, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 915, 919, 923, 924, 925,
         928, 930, 931],
        [439, 655, 733, 811, 848, 875, 881, 882, 889, 891, 894, 895, 896, 900, 908, 909, 910, 913, 915, 919, 923, 924,
         925],
        [439, 655, 848, 875, 881, 882, 889, 891, 894, 895, 896, 900, 909, 910, 913, 915, 919, 923, 924, 925, 926, 928,
         929, 932]]

    for data in dataset:
        data.sort()
        print(data)

    s_time = time()
    L = apriori(1, dataset)
    print(time()-s_time)
    result = []
    for _, Lk in L.items():
        result.extend(Lk.keys())
    print(result, len(result))
