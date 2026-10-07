from baseline.vsimserach.utils import compute_if_absent, get_from_multi_level_dict, init_bitset

class BaselineIndex:
  def build_index(self, nodes, edges):
    '''
    Index format:
      Node index: list of: type -> node_id -> # of out edges
      Node labels: type of: node_id -> type

      Edge index: list of: sid -> eid, theta, d
    '''
    grouped_nodes = nodes.groupby('frame')
    grouped_edges = edges.groupby('frame')
    node_index = dict()
    node_labels = dict()
    edge_index = dict()
    for fid in grouped_nodes.groups.keys():
      subtable_nodes = grouped_nodes.get_group(fid)
      subtable_edges = grouped_edges.get_group(fid)

      node_index_current_fid = compute_if_absent(node_index, fid, dict)
      edge_index_current_fid = compute_if_absent(edge_index, fid, dict)

      for _, node in subtable_nodes.iterrows():
        # init type info.
        node_type, node_id = int(node['class']), int(node['id'])
        existing_label = node_labels.get(node_id)
        if existing_label is None:
          node_labels[node_id] = node_type
        else:
          assert existing_label == node_type, \
            'node label conflict! stored label: {}, new label: {}'.format(existing_label, node_type)

      for _, edge in subtable_edges.iterrows():
        sid, eid, theta, d, fid = int(edge['sid']), int(edge['eid']), \
          edge['theta'], edge['d_ratio'], int(edge['frame'])
        # get type
        sid_type = node_labels[sid]
        # add to node
        sid_node_dict = compute_if_absent(node_index_current_fid, sid_type, dict)
        current_count = sid_node_dict.get(sid)
        if current_count is None:
          current_count = 0
        sid_node_dict[sid] = current_count + 1

        # add to edge
        sid_edge_list = compute_if_absent(edge_index_current_fid, sid, list)
        sid_edge_list.append((eid, theta, d))
    return node_index, edge_index, node_labels

class ProposedIndexPerFrame:

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

      # create type to type map
      current_frame_index = dict()
      for _, row in subtable_edges.iterrows():
        sid, eid, theta, d = int(row['sid']), int(row['eid']), \
          row['theta'], row['d_ratio']
        sid_type, eid_type = id_type_dict[sid], id_type_dict[eid]
        sid_type_index = compute_if_absent(current_frame_index, sid_type, dict)
        eid_type_index = compute_if_absent(sid_type_index, eid_type, dict)
        sid_eid_list = compute_if_absent(eid_type_index, (theta, d), list)
        sid_eid_list.append((sid, eid))
      index[fid] = current_frame_index
    return index

class ProposedIndex:
  def build_index(self, nodes, edges):
    '''
    Index format:
      stype -> etype -> theta, d -> sid, eid -> frame_list
    '''
    # TODO: build index based on class labels.
    # build edges.
    # sid, eid, \theta & d, flist
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
        # 将真实的d值作为匹配项
        # sid, eid, theta, d = int(row['sid']), int(row['eid']), \
        #   row['theta'], row['d']
        sid_type, eid_type = id_type_dict[sid], id_type_dict[eid]
        # create type to type map
        sid_type_index = compute_if_absent(index, sid_type, dict)
        eid_type_index = compute_if_absent(sid_type_index, eid_type, dict)
        edge_index = compute_if_absent(eid_type_index, (theta, d), dict)
        # frame_list = compute_if_absent(edge_index, (sid, eid), list)
        # frame_list.append(fid)
        frame_list = compute_if_absent(edge_index, (sid, eid), set)
        frame_list.add(fid)
    return index
