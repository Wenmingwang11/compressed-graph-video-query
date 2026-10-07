import pickle
def ca_window_count(video_name, s):
    windowid_index_path = f'./storage/index/{video_name}_{s}_windowid_index.pkl'
    with open(windowid_index_path, 'rb') as f:
        windowid_index = pickle.load(f)

    output_file = f'./storage/index/window_count/window_num_results.txt'
    with open(output_file, 'a') as f:
        f.write(f'{video_name}_{s}\n')
        f.write(f'window_id, node_num\n')
        f.write(f'{len(windowid_index)}\n')
        f.write('\n')







if __name__ == '__main__':
    videos = ['bdd100kA', 'bdd100kB', 'drtest', 'drtrain']
    # videos = ['drtrain']
    bili = [0.6, 0.7, 0.8, 0.9]
    # bili = [0.9]
    for v in videos:
        for i in bili:
            ca_window_count(v, i)
