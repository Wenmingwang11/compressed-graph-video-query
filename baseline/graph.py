import math
import pandas as pd
from vsimsearch.data import TemporalPattern

def compute_nodes_from_dataframe(df):
  max_frame = df['frame'].max()
  print('computing nodes from data frame, this could be very slow, please wait ...')
  # max_frame = 1
  # for each frame; compute a graph
  nodes = []
  for fid in range(1, max_frame+1):
    # select ids.
    current_frame_table = df[df['frame'] == fid]

    # collect points
    for _, row in current_frame_table.iterrows():
      _id = int(row['id'])
      _left, _top, _width, _height = row['left'], row['top'], row['width'], row['height']
      _point_x, _point_y = _left + _width / 2, _top + _height / 2
      nodes.append((fid, _id, _point_x, _point_y, _width, _height, row['class']))
      # nodes.append((fid, _id, _point_x, _point_y, _width, _height, row['class'], row['color']))

  node_pd = pd.DataFrame(nodes, columns=['frame', 'id', 'center_x', 'center_y', 'width', 'height', 'class'])
  # node_pd = pd.DataFrame(nodes, columns=['frame', 'id', 'center_x', 'center_y', 'width', 'height', 'class', 'color'])
  node_pd.astype({'frame': 'int', 'id': 'int'}, copy=False)
  print('node computation complete')
  return node_pd

def compute_edges_from_nodes(nodes, height, width, build_mode='normal'):
  '''
  build_mode: normal, complete or pattern.

  data frame format:
  frame, id, left, top, width, height, class_type
  '''
  base_distance = math.sqrt(width **2) + math.sqrt(height ** 2)

  max_frame = int(nodes['frame'].max())
  edges = list()
  for fid in range(1, max_frame + 1):
    id_pos_dict = {}
    # process this frame only
    nodes_in_fid = nodes[nodes['frame'] == fid]
    for _, row in nodes_in_fid.iterrows():
      _id, _point_x, _point_y = int(row['id']), row['center_x'], row['center_y']
      id_pos_dict[_id] = (_point_x, _point_y)

    if build_mode == 'normal':
      _build_directed_edges(fid, id_pos_dict, base_distance, edges)
    elif build_mode == 'complete':
      _build_bidirected_edges(fid, id_pos_dict, base_distance, edges)
    elif build_mode == 'pattern':
      _build_pattern_edges(fid, id_pos_dict, base_distance, edges)
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
