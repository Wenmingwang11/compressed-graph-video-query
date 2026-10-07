# for index building
import argparse
from graph import compute_edges_from_nodes
from baseline.vsimserach.io import read_nodes_from_mot_result, read_nodes_from_mot_result_multi_class
from common import add_args_for_discretize_func, index_types, discretize_func_from_args
import pickle
import time

def prepare_parser(parser: argparse.ArgumentParser):
  parser.add_argument('--file_path')
  parser.add_argument('--frame_height', type=int)
  parser.add_argument('--frame_width', type=int)
  parser.add_argument('--output_path')
  add_args_for_discretize_func(parser)
  parser.add_argument('--single_type', type=bool, default=False)
  parser.add_argument('--percent', type=float, default=-1)

if __name__ == '__main__':
  parser = argparse.ArgumentParser('build and persist index from a detected file')
  prepare_parser(parser)
  parser.add_argument('--index_type', type=int, choices=[1, 2, 3, 4])
  args = parser.parse_args()

  # start working
  nodes = None
  if args.single_type:
    nodes = read_nodes_from_mot_result(args.file_path)
  else:
    nodes = read_nodes_from_mot_result_multi_class(args.file_path)

  # percent
  if args.percent > 0:
    # filter nodes
    total_frames = nodes['frame'].max()
    assert args.percent <= 1
    stop_frame = total_frames * args.percent
    origin_len = len(nodes)
    nodes = nodes[nodes['frame'] <= stop_frame]
    print('node length', len(nodes), origin_len, len(nodes)/origin_len)
    print('frames', total_frames, stop_frame, stop_frame/total_frames)

  edges = compute_edges_from_nodes(nodes, args.frame_height, args.frame_width)
  # apply func
  discretize_func_from_args(args)(edges)

  index_builder = index_types[args.index_type]()

  start_time = time.process_time()
  index = index_builder.build_index(nodes, edges)
  end_time = time.process_time()
  print('time:', end_time-start_time)

  with open(args.output_path, 'wb') as wf:
    pickle.dump(index, wf)
