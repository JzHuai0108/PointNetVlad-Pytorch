import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KDTree

##########################################
# spilt the positive and negative samples of train data
# save in the training_queries_long_radar.pickle and training_queries_short_radar.pickle
##########################################

def construct_query_dict(df_centroids, seq_index, queries_len, positive_dist, negative_dist):
    tree = KDTree(df_centroids[['x', 'y']])
    ind_nn = tree.query_radius(df_centroids[['x', 'y']], r=positive_dist)
    ind_r = tree.query_radius(df_centroids[['x', 'y']], r=negative_dist)
    queries = {}
    for i in range(len(ind_nn)):
        data_id = int(df_centroids.iloc[i]["data_id"])
        data_file = df_centroids.iloc[i]["data_file"]
        yaw = df_centroids.iloc[i]["yaw"]
        positive_candis = np.setdiff1d(ind_nn[i], [i]).tolist()
        yaw_diff = np.abs(yaw - df_centroids.iloc[positive_candis]["yaw"])
        positives = [c for c in positive_candis if (yaw_diff[c] < cfgs.yaw_threshold) or (yaw_diff[c] > 360 - cfgs.yaw_threshold)]
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(), ind_r[i]).tolist()

        random.shuffle(negatives)
        if seq_index != 0:
            data_id += queries_len
            positives = [p + queries_len for p in positives]
            negatives = [n + queries_len for n in negatives]
        queries[data_id] = {"query": data_file, "positives": positives, "negatives": negatives}
    return queries

def split_dataset(base_path, data_type, save_path, positive_dist, negative_dist):
    data_path = os.path.join(base_path, data_type)
    groups = sorted(os.listdir(data_path))
    queries_len = 0
    train_seqs = {}
    for group_id, group in enumerate(tqdm(groups)):
        group_dir = os.path.join(data_path, group)
        seqs = [name for name in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, name))]
        df_train = pd.DataFrame(columns=['data_id', 'data_file', 'x', 'y', 'yaw'])
        for seq in tqdm(seqs):
            seq_poses_path = os.path.join(group_dir, seq + '_poses.txt')
            df_locations = pd.read_table(seq_poses_path, sep=' ', converters={'timestamp': str},
                                         names=['timestamp', 'r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z', 'yaw'])
            df_locations = df_locations.loc[:, ['timestamp', 'x', 'y', 'z', 'yaw']]
            df_locations['timestamp'] = data_type + '/' + group + '/' + seq + '/' + df_locations['timestamp'] + '.bin'
            df_locations = df_locations.rename(columns={'timestamp': 'data_file'})
            df_train = pd.concat([df_train, df_locations], ignore_index=True)
            df_train['data_id'] = range(0, len(df_train))

        queries = construct_query_dict(df_train, group_id, queries_len, positive_dist, negative_dist)
        train_seqs.update(queries)
        queries_len += len(queries)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as handle:
        pickle.dump(train_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/path/to/radar_reloc_data/', help='radar datasets path')
    parser.add_argument('--save_folder', type=str, default='./radar_split/', help='the saved path of split file ')
    parser.add_argument('--save_name', type=str, default='training_queries_short_radar.pickle',
                        help='saved file name, training_queries_short_radar.pickle/training_queries_long_radar.pickle')
    parser.add_argument('--data_type', type=str, default='train_short', help='train_short or train_long')
    parser.add_argument('--positive_dist', type=float, default=5, help='Positive sample distance threshold, short:5, long:10')
    parser.add_argument('--negative_dist', type=float, default=10, help='Negative sample distance threshold, short:9, long:18')
    parser.add_argument('--yaw_threshold', type=float, default=75, help='Yaw angle threshold')
    cfgs = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(cfgs.save_folder, cfgs.save_name)
    split_dataset(cfgs.data_path, cfgs.data_type, save_path, cfgs.positive_dist, cfgs.negative_dist)