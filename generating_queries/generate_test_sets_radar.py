import os
import pickle
import numpy as np
import pandas as pd
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
        database = {}
        query = {}
        for tra_id in range(len(tras)):
            pose_path = seq_path + '/' + tras[tra_id] + '_poses.txt'
            df_locations = pd.read_table(pose_path, sep=' ',
                                         names=['r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z', 'yaw'])
            df_locations['data_id'] = range(0, len(df_locations))
            df_locations = df_locations.loc[:, ['data_id', 'x', 'y', 'z', 'yaw']]
            df_locations['data_id'] = df_locations['data_id'].apply(lambda x: str(x).zfill(6))
            df_locations['data_id'] = data_type + '/' + seq + '/' + tras[tra_id] + '/' + df_locations['data_id'].astype(str) + '.bin'
            df_locations = df_locations.rename(columns={'data_id': 'data_file'})

            if tra_id == 0:
                df_database = df_locations
                database_tree = KDTree(df_database[['x', 'y']])
                database_trees.append(database_tree)
                for index, row in df_locations.iterrows():
                    database[len(database.keys())] = {
                        'query': row['data_file'], 'x': row['x'], 'y': row['y'], 'yaw': row['yaw']}
                database_sets[seq] = database
            else:
                df_test = df_locations
                for index, row in df_test.iterrows():
                    query[len(query.keys())] = {'query': row['data_file'], 'x': row['x'], 'y': row['y'], 'yaw': row['yaw']}
                query.update(query)
            query_sets[seq] = query

        for key in range(len(query_sets[seq].keys())):
            coor = np.array([[query_sets[seq][key]["x"], query_sets[seq][key]["y"]]])
            yaw = query_sets[seq][key]["yaw"]
            index = database_trees[seq_id].query_radius(coor, r=positive_dist)[0].tolist()
            yaw_diff = 120 - np.abs(yaw - df_database.iloc[index]["yaw"])
            true_index = np.array(index)[np.where(yaw_diff > 75)[0]]
            query_sets[seq][key][seq] = true_index.tolist()

    output_to_file(database_sets, save_folder + 'evaluation_database_' + data_type + '.pickle')
    output_to_file(query_sets, save_folder + 'evaluation_query_' + data_type + '_' + str(positive_dist) + 'm.pickle')

# Building database and query files for evaluation
if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = '/media/cyw/CYW-ZX2/radar_reloc_data/'
    save_folder = "./radar_split/"

    ######  for short range radar (coloradar, qinghua)  #####
    data_type = "test_short"
    seqs = ['ec_hallways', 'edgar_classroom', 'outdoors', 'seq3', 'seq4']
    positive_dist = 5
    construct_query_and_database_sets(base_path, data_type, seqs, positive_dist, save_folder)


    ######  for long range radar (inhouse, mscrad4r)  #####
    data_type = "test_long"
    seqs = ['RURAL_C', 'xinbu']
    positive_dist = 9
    construct_query_and_database_sets(base_path, data_type, seqs, positive_dist, save_folder)





