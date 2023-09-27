# -*-coding:utf-8-*-
import numpy as np
from tqdm import tqdm
import os
import argparse
from tools import random_down_sample

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
            yaws.append(yaw-180)
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
    return poses, yaws


def load_scans(scan_path):
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    current_points = current_vertex[:, 0:3]
    current_intensity = current_vertex[:, 3]
    current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points

    return current_vertex, current_intensity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/path/to/radar_reloc_data/',
                        help='radar datasets path')
    parser.add_argument('--dataset_name', type=str, default='qinghua', help='radar dataset folder')
    parser.add_argument('--seqs', type=list, default=['seq1', 'seq2','seq3', 'seq4', 'seq5', 'seq7'],
                        help='the groups name in the dataset')
    parser.add_argument('--train_seqs', type=list, default=[1, 2, 5, 7], help='train seqs in the dataset')
    parser.add_argument('--train_folder', type=str, default='train_short', help='train submaps saved folder')
    parser.add_argument('--test_folder', type=str, default='test_short', help='test submaps saved folder')
    parser.add_argument('--frame_winsize', type=int, default=5, help='window size for submap')
    parser.add_argument('--target_points', type=int, default=512, help='traget points in saved submaps')
    cfgs = parser.parse_args()

    dataset_path = os.path.join(cfgs.data_path, cfgs.dataset_name)
    calib_path = os.path.join(dataset_path, 'calib/base_to_single_chip.txt')

    for seq in tqdm(cfgs.seqs):
        seq_id = int(seq[-1])
        tra_folders = (os.listdir(os.path.join(dataset_path, seq)))
        for tra in tra_folders:
            tra_id = tra.split('-')[-1]

            if seq_id in cfgs.train_seqs:
                save_folder = cfgs.train_folder
                gap_size = cfgs.frame_winsize // 2
            else:
                save_folder = cfgs.test_folder
                if tra_id == '1':
                    gap_size = cfgs.frame_winsize // 2
                else:
                    gap_size = cfgs.frame_winsize

            save_dir = os.path.join(cfgs.data_path, save_folder, seq, tra)
            os.makedirs(save_dir, exist_ok=True)

            tra_path = os.path.join(dataset_path, seq, tra)
            scans_path = tra_path + "/bins"
            xyz_path = tra_path + '/data_label.txt'
            yaw_path = tra_path + '/yaw.txt'
            poses, yaws = load_qinghua_poses(xyz_path, yaw_path)
            file_names = sorted(os.listdir(scans_path))

            submap_poses = []
            submap_yaws = []
            count = 0
            for i in range(0, len(file_names), gap_size):
                end = i + cfgs.frame_winsize
                if end > len(file_names):
                    continue
                submap_pc = np.empty((0,0), dtype=float, order='C')
                submap_pose = poses[i + cfgs.frame_winsize//2]
                submap_yaw = yaws[i + cfgs.frame_winsize//2]
                for j in range(i, end):
                    if j < len(file_names):
                        neiscan_pc, neiscan_intensity = load_scans(
                            os.path.join(scans_path, file_names[j]))  # near neighbour scan point clouds
                        Rc_T_Rj = np.matmul(np.linalg.inv(poses[i]), poses[j])
                        temp_pc_in_center = np.matmul(Rc_T_Rj, neiscan_pc.T).T
                        submap_pc = np.concatenate((submap_pc, temp_pc_in_center), axis=0) if submap_pc.size else temp_pc_in_center

                target_submap = random_down_sample(submap_pc[:, :3], cfgs.target_points)

                with open(os.path.join(save_dir, (str(count).zfill(6) + ".bin")), 'wb') as f:
                    target_submap.tofile(f)
                submap_poses.append(submap_pose)
                submap_yaws.append(submap_yaw)
                count += 1

            submaps_poses_path = save_dir + '_poses.txt'
            with open(submaps_poses_path, 'w', encoding='utf-8') as f:
                for pose_id, pose in enumerate(submap_poses):
                    pose_reshape = pose[:3, :4].reshape(1, 12).flatten()
                    time_i = [str(pose_id).zfill(6)]
                    yaw_i = np.array([submap_yaws[pose_id]])
                    pose_with_yaw = np.concatenate((time_i, pose_reshape, yaw_i))
                    f.write(' '.join(str(x) for x in pose_with_yaw) + '\n')