from bitarray import bitarray
import math
from collections import defaultdict
def compute_if_absent(some_dict, key, default_value_func):
  key_value = some_dict.get(key)
  if key_value is None:
    key_value = default_value_func()
    some_dict[key] = key_value
  return key_value

def get_from_multi_level_dict(some_dict, keys):
  current_dict = some_dict
  for key in keys:
    current_dict = current_dict.get(key)
    if current_dict is None:
      return None
  return current_dict

def get_from_multi_level_dict2(some_dict, keys):
  current_dict = some_dict
  for i, key in enumerate(keys):
    current_dict = current_dict.get(key)
    if current_dict is None and i == 1 and isinstance(keys[i], tuple):
      # Try to find keys within a range of 1 from each value in the tuple
      adjacent_dicts = []
      for adj_key in keys[i]:
        for offset in range(-1, 2):  # Check adjacent keys with error of 1
          adjacent_key = adj_key + offset
          adjacent_dict = some_dict.get(adjacent_key)
          if adjacent_dict is not None:
            adjacent_dicts.append(adjacent_dict)
      if adjacent_dicts:
        return adjacent_dicts
    if current_dict is None:
      return None
  return current_dict


def merge_dicts(dict1, dict2):
  if not dict1:
    return dict2
  else:
    result = dict1.copy()
    for key, value2 in dict2.items():
      if key in result:
        value1 = result[key]
        if isinstance(value1, set):
          result[key] = value1 | value2  # 合并两个集合
        else:
          result[key] = {value1, value2}  # 创建一个新集合，将两个值加入
      else:
        result[key] = value2
    return result

# def merge_dicts(dict1, dict2):
#   if not isinstance(dict1, dict) or not isinstance(dict2, dict):
#     raise TypeError("Both arguments must be dictionaries")
#
#   merged_dict = defaultdict(set)
#
#   # Add values from dict1
#   for key, value in dict1.items():
#     merged_dict[key].add(value)
#
#     # Add values from dict2
#   for key, value in dict2.items():
#     merged_dict[key].add(value)
#
#   # Convert sets to single values if possible
#   result = {}
#   for key, values in merged_dict.items():
#     if len(values) == 1:
#       result[key] = values.pop()
#     else:
#       result[key] = values
#
#   return result

def get_from_multi_level_dict_with_tolerance(some_dict, keys):
  # theta_tolerance = 0.2 // (math.pi / 360)
  # d_tolerance = 0.2 // (1 / 2000)
  theta_tolerance = 0.12
  d_tolerance = 0.8
  current_dict = some_dict
  for key in keys:
    if isinstance(key, tuple) and len(key) == 2:
      # Handle (theta, d) key
      theta, d = key
      sub_matchs = {}
      for sub_key, value in current_dict.items():
        sub_theta, sub_d = sub_key
        if abs(sub_theta - theta) <= theta_tolerance and abs(abs(sub_d) - abs(d)) <= d_tolerance:
          sub_matchs = merge_dicts(sub_matchs, value)
      if sub_matchs is None:
        return None
      current_dict = sub_matchs
      break
    else:
      current_dict = current_dict.get(key)
      if current_dict is None:
        return None
  return current_dict


# def matching_with_calculate_score(some_dict, keys):
#   theta_tolerance = 0.08 // (math.pi / 360)
#   d_tolerance = 0.05 // (1 / 2000)
#   # theta_tolerance = 0
#   # d_tolerance = 0
#   epsilon = 1e-10
#   current_dict = some_dict
#   for key in keys:
#     if isinstance(key, tuple) and len(key) == 2:
#       # Handle (theta, d) key
#       theta, d = key
#       sub_matchs = {}
#       for sub_key, value in current_dict.items():
#         sub_theta, sub_d = sub_key
#         if abs(sub_theta - theta) <= theta_tolerance and abs(abs(sub_d) - abs(d)) <= d_tolerance:
#           # Calculate e_score
#           X = theta
#           Y = sub_theta
#           e_score = 1 - abs((Y - X) / (X + epsilon))
#           sub_matchs[e_score] = value
#           # sub_matchs = merge_dicts(sub_matchs, )
#       if sub_matchs is None:
#         return None
#       current_dict = sub_matchs
#       break
#     else:
#       current_dict = current_dict.get(key)
#       if current_dict is None:
#         return None
#   return current_dict


def init_bitset(length):
  bitset = bitarray(length)
  bitset.setall(0)
  return bitset


def print_multi_level_dict(some_dict, padding=0):
  for key, value in some_dict.items():
    print(' '*padding + str(key))
    if type(value) == dict:
      print_multi_level_dict(value, padding + 2)
    else:
      print(' '*(padding+2) + str(value))

def discretize_attributes(df, discretize_columns):
  for column, func in discretize_columns:
    df[column] = df[column].apply(func)

def discretize_function_4(edges, theta_n_parts=10, theta_d_parts=8):
  discretize_attributes(edges, [
    ('theta', lambda x: x // (math.pi / theta_n_parts)),
    ('d_ratio', lambda x: x // (1 / theta_d_parts))
  ])
