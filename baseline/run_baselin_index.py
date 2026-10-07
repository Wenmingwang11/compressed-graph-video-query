from settings import raw_file_path, index_file_path, video_name_meta_mapping, df_funcs
import os
# pass

def build_with_index_for(video_name, index_type, df_name, percent=None):
  width, height = video_name_meta_mapping[video_name]
  file_path = '{}/{}.txt'.format(raw_file_path, video_name)
  df, df_p1, df_p2 = df_funcs[df_name]
  if percent is None:
    output_path = '{}/{}-{}-{}.pkl'.format(index_file_path, \
      video_name, index_type, df_name)
    log_path = '{}/{}-{}-{}.out.txt'.format(index_file_path, \
      video_name, index_type, df_name)
  else:
    output_path = '{}/percent/{}-{}-{}-{}.pkl'.format(index_file_path, \
      video_name, index_type, df_name, percent)
    log_path = '{}/percent/{}-{}-{}-{}.out.txt'.format(index_file_path, \
      video_name, index_type, df_name, percent)
  tokens = ['python', 'baseline_index_app.py',
    '--file_path', file_path,
    '--frame_height', height, '--frame_width', width,
    '--output_path', output_path,
    '--index_type', index_type,
    '--discretize_func', df,
    '--df_param1', df_p1, '--df_param2', df_p2
  ]
  if percent is not None:
    tokens.extend(['--percent', str(percent)])
  # mkdir if needed
  if not os.path.isdir(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))
  command = ' '.join([str(t) for t in tokens]) + ' > ' +  log_path
  print(command)
  os.system(command)

if __name__ == '__main__':


  # dfs = ['df5','df6', 'df7', 'df8']
  # for df in dfs:
  #   build_with_index_for('drtrain', 1, df)
  #   build_with_index_for('drtest', 1, df)
  #   # build_with_index_for('bdd100kA', 1, df)
  #   # build_with_index_for('bdd100kB', 1, df)

  # videos = ['drtrain', 'drtest', 'bdd100kA', 'bdd100kB']
  videos = ['drtest']

  for v in videos:
    for i in [1]:
      build_with_index_for(v, i, 'df0')
