
# file_storage_path = '../storage/results/vsim'
# file_storage_path = '../storage/results/vsim-detrac'
# file_storage_path = '../storage/results/vsim-detrac-10'
# file_storage_path = '../storage/results/vsim-detrac-5'
# file_storage_path = '../storage/results/vsim-detrac-5-fixed'

file_storage_path = '../storage/results_new/vsim-all'
multi_file_storage_path = '../../storage/results_new/vsim-all'

raw_file_path = '../storage'
index_file_path = './storage/baseline_index'
# pattern_file_path = file_storage_path+'/pattern'
pattern_file_path = file_storage_path+'/batch_patterns'
result_file_path = file_storage_path+'/result'
seq_index_file_path = file_storage_path+'/seq_index'

inspectors_file_path = file_storage_path+ '/inspectors'

_720p = (1280, 720)
_1280p = (1920, 1080)
_qHD = (960, 540)

video_name_meta_mapping = dict({
  # name: (width, height)
  'test': _1280p,
  'traffic1': _720p,
  'traffic2': _720p,
  'movie-ff': _720p,
  'drtrain': _qHD,
  'drtest': _qHD,
  'bdd100kA': _720p,
  'bdd100kB': _720p,
  'I-24': _1280p,
  'color_drtrain': _qHD,
  'sampling-drtrain': _qHD,
  'ADL-Rundle': _qHD,
  'sampling-ADL_Rundle': _qHD,
  'MOT17_test': _1280p,
  'MOT17_train': _1280p,
  'MOT16_train': _1280p,
  'didi1003': _1280p
})

df_funcs = dict({
  # name: (discretize_func, df_param1, df_param2)
  'df0': (4, 1, 1),
  'df1': (4, 2, 3),
  'df2': (4, 2, 5),
  'df3': (4, 4, 3),
  'df4': (4, 4, 5),
  'df5': (4, 4, 10),
  'df6': (4, 8, 10),
  'df7': (4, 8, 15),
  'df8': (4, 12, 15),
  'df9': (1, 4, 4),
  'df10': (1, 3, 3),
  'df11': (4, 90, 1000),
  'df12': (4, 180, 1200),
  'df13': (4, 360, 2000)
})
