import os
import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KDTree

##########################################
# spilt the positive and negative samples of train data
# save in the training_queries_baseline.pickle
##########################################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = '/path/to/radar_reloc_data'

def construct_query_dict(df_centroids, seq_index, queries_len, positive_dis, negative_dis):
    tree = KDTree(df_centroids[['x', 'y']])
    ind_nn = tree.query_radius(df_centroids[['x', 'y']], r=positive_dis)
    ind_r = tree.query_radius(df_centroids[['x', 'y']], r=negative_dis)
    queries = {}
    for i in range(len(ind_nn)):
        data_id = int(df_centroids.iloc[i]["data_id"])
        data_file = df_centroids.iloc[i]["data_file"]
        yaw = df_centroids.iloc[i]["yaw"]
        positive_candis = np.setdiff1d(ind_nn[i], [i]).tolist()
        yaw_diff = 120 - np.abs(yaw - df_centroids.iloc[positive_candis]["yaw"])
        positives = np.array(positive_candis)[np.where(yaw_diff > 75)[0]].tolist()
        negatives = np.setdiff1d(
            df_centroids.index.values.tolist(), ind_r[i]).tolist()

        random.shuffle(negatives)
        if seq_index != 0:
            data_id += queries_len
            positives = [p + queries_len for p in positives]
            negatives = [n + queries_len for n in negatives]
        queries[data_id] = {"query": data_file, "positives": positives, "negatives": negatives}
    return queries

def split_dataset(base_path, data_type, save_path, positive_dis, negative_dis):
    data_path = os.path.join(base_path, data_type)
    groups = sorted(os.listdir(data_path))  # 所有组数据
    queries_len = 0
    train_seqs = {}
    for group_id, group in enumerate(tqdm(groups)):
        group_dir = os.path.join(data_path, group)
        seqs = [name for name in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, name))]
        df_train = pd.DataFrame(columns=['data_id', 'data_file', 'x', 'y', 'yaw'])
        for seq in tqdm(seqs):
            seq_poses_path = os.path.join(group_dir, seq + '_poses.txt')
            df_locations = pd.read_table(seq_poses_path, sep=' ',
                                         names=['r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33',
                                                'z', 'yaw'])

            df_locations['data_file'] = range(0, len(df_locations))
            df_locations = df_locations.loc[:, ['data_file', 'x', 'y', 'z', 'yaw']]
            df_locations['data_file'] = df_locations['data_file'].apply(lambda x: str(x).zfill(6))
            df_locations['data_file'] = data_type + '/' + group + '/' + seq + '/' + df_locations['data_file'].astype(str) + '.bin'
            df_train = pd.concat([df_train, df_locations], ignore_index=True)
            df_train['data_id'] = range(0, len(df_train))

        queries = construct_query_dict(df_train, group_id, queries_len, positive_dis, negative_dis)
        train_seqs.update(queries)
        queries_len += len(queries)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as handle:
        pickle.dump(train_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)

##### for short range radar (coloradar, qinghua) ###
data_type = 'train_short'
save_path = "./radar_split/training_queries_short_radar.pickle"
positive_dis = 5
negative_dis = 10
split_dataset(base_path, data_type, save_path, positive_dis, negative_dis)

#### for long range radar (inhouse, mscrad4r) ###
data_type = 'train_long'
save_path = "./radar_split/training_queries_long_radar.pickle"
positive_dis = 9
negative_dis = 18
split_dataset(base_path, data_type, save_path, positive_dis, negative_dis)