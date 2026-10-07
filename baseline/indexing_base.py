from baseline.vsimserach.utils import compute_if_absent

class VertexOnlyIndex:

  def build_index(self, nodes, edges):
    grouped_nodes = nodes.groupby('frame')
    grouped_edges = edges.groupby('frame')
    index = dict()

    for fid in grouped_edges.groups.keys():
      subtable_nodes = grouped_nodes.get_group(fid)
      subtable_edges = grouped_edges.get_group(fid)

      # collect id-> type
      id_type_dict = dict()
      for _, node in subtable_nodes.iterrows():
        node_id, node_type = int(node['id']), int(node['class'])
        id_type_dict[node_id] = node_type

      for _, row in subtable_edges.iterrows():
        sid, eid, theta, d = int(row['sid']), int(row['eid']), \
          row['theta'], row['d_ratio']
        sid_type, eid_type = id_type_dict[sid], id_type_dict[eid]
        # create type to type map
        data_list = compute_if_absent(index, (sid_type, eid_type), dict)
        frame_list = compute_if_absent(data_list, (sid, eid, theta, d), list)
        frame_list.append(fid)
    return index

class EdgeOnlyIndex:

  def build_index(self, nodes, edges):
    grouped_nodes = nodes.groupby('frame')
    grouped_edges = edges.groupby('frame')
    index = dict()

    for fid in grouped_edges.groups.keys():
      subtable_nodes = grouped_nodes.get_group(fid)
      subtable_edges = grouped_edges.get_group(fid)

      # collect id-> type
      id_type_dict = dict()
      for _, node in subtable_nodes.iterrows():
        node_id, node_type = int(node['id']), int(node['class'])
        id_type_dict[node_id] = node_type

      for _, row in subtable_edges.iterrows():
        sid, eid, theta, d = int(row['sid']), int(row['eid']), \
          row['theta'], row['d_ratio']
        sid_type, eid_type = id_type_dict[sid], id_type_dict[eid]
        # create type to type map
        data_level = compute_if_absent(index, (theta, d), dict)
        frame_list = compute_if_absent(data_level, (sid, eid, sid_type, eid_type), list)
        frame_list.append(fid)
    return index
