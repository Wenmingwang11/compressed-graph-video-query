import math
import pandas as pd
from vsimsearch.graph import compute_nodes_from_dataframe
import os, pickle
'''
Return two tables:

---
vertex table:
frame | vertex_id | pos_x | pos_y | width | height | class | ...other_attributes
...

---
edge table:
frame | start_vertex | end_vertex | theta | d |
'''

def _read_nodes_from_mot_result(file_path, class_type=1):
  df = pd.read_csv(file_path, names=[
    'frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'x', 'y', 'z'
  ], index_col=False)
  # objects' position can not exceed image boundary
  # add column
  df['class'] = [class_type] * len(df.index)
  df.drop(columns=['conf', 'x', 'y', 'z'])
  return compute_nodes_from_dataframe(df)

def _cache_exists(file_path):
  cache_path = file_path+'.cache.pkl'
  return os.path.isfile(cache_path)

def _load_from_cache(file_path):
  cache_path = file_path+'.cache.pkl'
  with open(cache_path, 'rb') as rf:
    return pickle.load(rf)

def _dump_to_cache(file_path, nodes):
  print('creating cache for further use ...')
  cache_path = file_path+'.cache.pkl'
  if not os.path.isdir(os.path.dirname(cache_path)):
    os.makedirs(os.path.dirname(cache_path))
  with open(cache_path, 'wb') as wf:
    pickle.dump(nodes, wf)
  print('cache done.')


def read_nodes_from_mot_result(file_path, class_type=1, enable_cache=True):
  if enable_cache and _cache_exists(file_path):
    return _load_from_cache(file_path)
  else:
    res = _read_nodes_from_mot_result(file_path, class_type)
    _dump_to_cache(file_path, res)
    return res

def _read_nodes_from_mot_result_multi_class(file_path):
  print('reading values from raw data, this could be slow, please wait ...')
  df = pd.read_csv(file_path, names=[
    'frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'class', 'x', 'y', 'z'
  ], index_col=False)
  # objects' position can not exceed image boundary
  # add column
  df.drop(columns=['conf', 'x', 'y', 'z'])
  return compute_nodes_from_dataframe(df)

# def _read_nodes_from_mot_result_multi_class(file_path):
#   print('reading values from raw data, this could be slow, please wait ...')
#   df = pd.read_csv(file_path, names=[
#     'frame', 'id', 'left', 'top', 'width', 'height', 'class'
#   ], index_col=False)
#   # df = pd.read_csv(file_path, names=[
#   #   'frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'class', 'x', 'y', 'z', 'color'
#   # ], index_col=False)
#   # objects' position can not exceed image boundary
#   # add column
#   # df.drop(columns=['conf', 'x', 'y', 'z'])
#   return compute_nodes_from_dataframe(df)

def read_nodes_from_mot_result_multi_class(file_path, enable_cache=True):
  if enable_cache and _cache_exists(file_path):
    return _load_from_cache(file_path)
  else:
    res = _read_nodes_from_mot_result_multi_class(file_path)
    _dump_to_cache(file_path, res)
    return res

if __name__ == '__main__':
  df = read_nodes_from_mot_result_multi_class('../storage/results_new/vsim-all/raw/traffic2.txt')
  print(df)
