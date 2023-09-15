import os
import pickle
import numpy as np
import pandas as pd
import argparse
import tqdm
from sklearn.neighbors import KDTree

##########################################
# split query and database data
# save in evaluation_database.pickle / evaluation_query.pickle
##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, data_type, seqs, positive_dist, save_folder):
    database_trees = []
    database_sets = {}
    query_sets = {}

    for seq_id, seq in enumerate(tqdm.tqdm(seqs)):
        seq_path = base_path + data_type + '/' + seq
        tras = [name for name in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, name))]
        query = {}
        for tra_id in range(len(tras)):
            pose_path = seq_path + '/' + tras[tra_id] + '_poses.txt'
            df_locations = pd.read_table(pose_path, sep=' ', converters={'timestamp': str},
                                         names=['timestamp','r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z', 'yaw'])
            df_locations = df_locations.loc[:, ['timestamp', 'x', 'y', 'z', 'yaw']]
            df_locations['timestamp'] = data_type + '/' + seq + '/' + tras[tra_id] + '/' + df_locations['timestamp'] + '.bin'
            df_locations = df_locations.rename(columns={'timestamp': 'data_file'})

            if tra_id == 0:
                df_database = df_locations
                database_tree = KDTree(df_database[['x', 'y']])
                database_trees.append(database_tree)
                database = {}
                for index, row in df_locations.iterrows():
                    database[len(database.keys())] = {
                        'query': row['data_file'], 'x': row['x'], 'y': row['y'], 'yaw': row['yaw']}
                database_sets[seq] = database
            else:
                df_test = df_locations
                for index, row in df_test.iterrows():
                    query[len(query.keys())] = {'query': row['data_file'], 'x': row['x'], 'y': row['y'], 'yaw': row['yaw']}
        query_sets[seq] = query

        for key in range(len(query_sets[seq].keys())):
            coor = np.array([[query_sets[seq][key]["x"], query_sets[seq][key]["y"]]])
            yaw = query_sets[seq][key]["yaw"]
            index = database_trees[seq_id].query_radius(coor, r=positive_dist)[0].tolist()
            yaw_diff = np.abs(yaw - df_database.iloc[index]["yaw"])
            true_index = [c for c in index if (yaw_diff[c] < cfgs.yaw_threshold) or (yaw_diff[c] > 360 - cfgs.yaw_threshold)]
            query_sets[seq][key][seq] = true_index

    output_to_file(database_sets, save_folder + 'evaluation_database_' + data_type + '.pickle')
    output_to_file(query_sets, save_folder + 'evaluation_query_' + data_type + '_' + str(positive_dist) + 'm.pickle')

# Building database and query files for evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/path/to/radar_reloc_data/',
                        help='radar datasets path')
    parser.add_argument('--save_folder', type=str, default='./radar_split/', help='the saved path of split file ')
    parser.add_argument('--data_type', type=str, default='test_short', help='test_short or test_long')
    parser.add_argument('--positive_dist', type=float, default=5,
                        help='Positive sample distance threshold, short:5, long:10')
    parser.add_argument('--yaw_threshold', type=float, default=75, help='Yaw angle threshold')
    cfgs = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if cfgs.data_type == "test_short":
        seqs = ['ec_hallways', 'edgar_classroom', 'outdoors', 'seq3', 'seq4']
    else:
        seqs = ['RURAL_C', 'xinbu']

    construct_query_and_database_sets(cfgs.data_path, cfgs.data_type, seqs, cfgs.positive_dist, cfgs.save_folder)







