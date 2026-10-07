import math
import pandas as pd
from vsimsearch.data import TemporalPattern

def compute_nodes_from_dataframe(df):
  max_frame = df['frame'].max()
  print('computing nodes from data frame, this could be very slow, please wait ...')
  nodes = []
  frame_ids_lists = []
  grouped_frames = {int(fid): frame_df for fid, frame_df in df.groupby('frame', sort=True)}

  for fid in range(1, max_frame+1):
    current_frame_table = grouped_frames.get(fid)
    if current_frame_table is None:
      frame_ids_lists.append([])
      continue

    frame_ids_list = current_frame_table['id'].tolist()
    frame_ids_lists.append(frame_ids_list)

    for row in current_frame_table.itertuples(index=False, name=None):
      _, _id, _left, _top, _width, _height, _class = row
      _point_x, _point_y = _left + _width / 2, _top + _height / 2
      nodes.append((fid, int(_id), _point_x, _point_y, _width, _height, _class))

  node_pd = pd.DataFrame(nodes, columns=['frame', 'id', 'center_x', 'center_y', 'width', 'height', 'class'])
  node_pd.astype({'frame': 'int', 'id': 'int'}, copy=False)
  print('node computation complete')
  return node_pd, frame_ids_lists

def compute_edges_from_nodes(nodes, height, width, build_mode='normal'):
  '''
  build_mode: normal, complete or pattern.

  data frame format:
  frame, id, left, top, width, height, class_type
  '''
  base_distance = math.sqrt(width **2) + math.sqrt(height ** 2)
  edges = list()

  for fid, nodes_in_fid in nodes.groupby('frame', sort=True):
    id_pos_dict = {
      int(row[1]): (row[2], row[3])
      for row in nodes_in_fid.itertuples(index=False, name=None)
    }

    if build_mode == 'normal':
      _build_directed_edges(int(fid), id_pos_dict, base_distance, edges)
    elif build_mode == 'complete':
      _build_bidirected_edges(int(fid), id_pos_dict, base_distance, edges)
    elif build_mode == 'pattern':
      _build_pattern_edges(int(fid), id_pos_dict, base_distance, edges)
    else:
      assert False, 'build mode can only be normal, complete or pattern'

  edge_pd = pd.DataFrame(edges, columns=['frame', 'sid', 'eid', 'theta', 'd', 'd_ratio'])
  edge_pd.astype({'frame': 'int', 'sid': 'int', 'eid': 'int'}, copy=False)
  # print(node_pd.dtypes)
  return edge_pd

def _build_edge_between(id_pos_item1, id_pos_item2, fid, base_distance, edges):
  id1, (x1, y1) = id_pos_item1
  id2, (x2, y2) = id_pos_item2
  _theta = math.atan2(y2-y1, x2-x1)
  _distance = math.sqrt((y2-y1) **2 + (x2-x1) ** 2)
  _distance_ratio = _distance/base_distance
  edges.append((fid, id1, id2, _theta, _distance, _distance_ratio))

def _build_pattern_edges(fid, id_pos_dict, base_distance, edges):
  sorted_items = sorted(id_pos_dict.items(), key=lambda x: x[1])
  # select the first one as the anchor node
  if len(sorted_items) <= 1:
    return
  for item2 in sorted_items[1:]:
    _build_edge_between(sorted_items[0], item2, fid, base_distance, edges)

def _build_directed_edges(fid, id_pos_dict, base_distance, edges):
  # only build if x1 < x2
  sorted_items = sorted(id_pos_dict.items(), key=lambda x: x[1])
  for _idx1 in range(len(sorted_items)):
    for _idx2 in range(_idx1 + 1, len(sorted_items)):
      _build_edge_between(sorted_items[_idx1], sorted_items[_idx2], fid, base_distance, edges)

def _build_bidirected_edges(fid, id_pos_dict, base_distance, edges):
    # compute edges.
    for id1, (x1, y1) in id_pos_dict.items():
      for id2, (x2, y2) in id_pos_dict.items():
        if id1 == id2:
          continue
        _build_edge_between((id1, (x1, y1)), (id2, (x2, y2)), fid, base_distance, edges)


def extract_pattern_graph(nodes, width, height, ids, frame_tuple):
  start_frame_inclusive, end_frame_inclusive = frame_tuple
  # print(nodes[nodes['id'].isin(ids)])
  filtered_nodes = nodes[
        (nodes['id'].isin(ids)) & \
        (nodes['frame'] >= start_frame_inclusive) & \
        (nodes['frame'] <= end_frame_inclusive)
      ]
  # update frame, starts with 1
  reset_frame_lambda = lambda x : x - start_frame_inclusive + 1
  # filtered_nodes['frame'] = filtered_nodes['frame'].apply(reset_frame_lambda)
  filtered_nodes.loc[:, 'frame'] = filtered_nodes['frame'].apply(reset_frame_lambda)
  # print('filtered_nodes', filtered_nodes)

  edges = compute_edges_from_nodes(filtered_nodes, height, width, 'pattern')

  return TemporalPattern(filtered_nodes, edges, \
    end_frame_inclusive-start_frame_inclusive + 1)
