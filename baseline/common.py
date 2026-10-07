from indexing import ProposedIndex, ProposedIndexPerFrame
from indexing_base import EdgeOnlyIndex, VertexOnlyIndex
from preprocessing import discretize_function_1, discretize_function_2, discretize_function_3, discretize_function_4
import argparse

index_types = dict([
  (1, ProposedIndex),
  (2, ProposedIndexPerFrame),
  (3, VertexOnlyIndex),
  (4, EdgeOnlyIndex),
])

def add_args_for_discretize_func(args: argparse.ArgumentParser):
  args.add_argument('--discretize_func', type=int)
  # add two funcs
  args.add_argument('--df_param1', type=int)
  args.add_argument('--df_param2', type=int)

def discretize_func_from_args(args: argparse.Namespace):
  func_id = args.discretize_func
  param1 = args.df_param1
  param2 = args.df_param2

  if func_id == 1:
    return lambda x: discretize_function_1(x, param1, param2)
  elif func_id == 2:
    return discretize_function_2
  elif func_id == 3:
    return lambda x: discretize_function_3(x, param1)
  elif func_id == 4:
    return lambda x:  discretize_function_4(x, param1, param2)
  else:
    assert False, 'Expecting func_id \in [1,4]'
