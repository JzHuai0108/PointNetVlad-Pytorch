# -*-coding:utf-8-*-
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import random
import os
import rosbag
import sensor_msgs.point_cloud2 as pc2
from test_slerp import *
import open3d as o3d
import re

def load_qinghua_poses(xyz_path, yaw_path):
    trans = []
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            xyz = np.fromstring(line, dtype=np.float32, sep=' ')
            xyz = xyz.reshape(3, 1)
            trans.append(xyz)

    poses = []
    with open(yaw_path, 'r') as f:
        yaws = []
        lines_yaw = f.readlines()
        for i in range(len(lines_yaw)):
            yaw = np.fromstring(lines_yaw[i], dtype=np.float32, sep=' ')[0]
            yaw = -(np.pi / 180) * yaw  # 将yaw角度转为弧度
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0],
                                        [sin_yaw, cos_yaw, 0],
                                        [0, 0, 1]])  # 旋转矩阵
            xyz = trans[i]
            pose = np.hstack((rotation_matrix, xyz))  # 将xyz和rotation矩阵按列拼接
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
            yaws.append(yaw)
    return poses, yaws


def random_down_sample(pc, sample_points):
    submap_pcd = o3d.geometry.PointCloud()
    submap_pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    sampleA = random.sample(range(pc.shape[0]), sample_points)
    sampled_cloud = submap_pcd.select_by_index(sampleA)
    sampled_pc = np.array(sampled_cloud.points)
    return sampled_pc


def load_scans(scan_path):
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    current_points = current_vertex[:, 0:3]
    current_intensity = current_vertex[:, 3]
    current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points

    return current_vertex, current_intensity


if __name__ == '__main__':
    data_path = '/path/to/radar_reloc_data/qinghua/'
    save_path = '/path/to/radar_reloc_data/'
    train_folder = 'train_short'
    test_folder = 'test_short'
    sub_size = 5
    target_points = 512
    seqs = ['seq1', 'seq2','seq3', 'seq4', 'seq5', 'seq7']
    train_id = [1, 2, 5, 7]

    for seq in tqdm(seqs):
        seq_id = int(seq[-1])
        tra_folders = (os.listdir(os.path.join(data_path, seq)))
        for tra in tra_folders:
            tra_id = tra.split('-')[-1]

            if seq_id in train_id:
                save_folder = train_folder
                gap_size = sub_size // 2
            else:
                save_folder = test_folder
                if tra_id == '1':
                    gap_size = sub_size // 2
                else:
                    gap_size = sub_size

            tra_path = data_path + seq + "/" + tra
            scans_path = tra_path + "/bins"
            xyz_path = tra_path + '/data_label.txt'
            yaw_path = tra_path + '/yaw.txt'
            poses, yaws = load_qinghua_poses(xyz_path, yaw_path)
            file_names = sorted(os.listdir(scans_path))

            submap_poses = []
            count = 0
            for i in range(0, len(file_names), gap_size):
                end = i + sub_size
                if end > len(file_names):
                    continue
                submap_pc = np.empty((0,0), dtype=float, order='C')
                submap_pose = poses[i+sub_size//2]
                for j in range(i, end):
                    if j < len(file_names):
                        neiscan_pc, neiscan_intensity = load_scans(
                            os.path.join(scans_path, file_names[j]))  # near neighbour scan point clouds
                        relative_pose = np.linalg.inv(poses[i]).dot(poses[j])
                        temp_pc_in_center = np.linalg.inv(poses[i]).dot(poses[j]).dot(neiscan_pc.T).T
                        submap_pc = np.concatenate((submap_pc, temp_pc_in_center), axis=0) if submap_pc.size else temp_pc_in_center
                
                submap_pc = submap_pc[np.linalg.norm(submap_pc[:, :3], axis=1) <= 200]
                target_submap = random_down_sample(submap_pc[:, :3], target_points)

                save_dir = os.path.join(save_path, save_folder, seq, tra)
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, (str(count).zfill(6) + ".bin")), 'wb') as f:
                    target_submap.tofile(f)
                submap_poses.append(submap_pose)
                count += 1

            submaps_poses_path = os.path.join(save_path, save_folder, seq, tra + '_poses.txt')
            with open(submaps_poses_path, 'w', encoding='utf-8') as f:
                for pose_id, pose in enumerate(submap_poses):
                    pose_reshape = pose[:3, :4].reshape(1, 12).flatten()
                    yaw_i = np.array([yaws[pose_id]])
                    pose_with_yaw = np.concatenate((pose_reshape, yaw_i))
                    f.write(' '.join(str(x) for x in pose_with_yaw) + '\n')