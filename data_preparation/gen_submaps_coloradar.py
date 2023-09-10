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

def interpolation(gt_times, gt_positions, gt_quats, radar_times):
    interp_positions = pos_interpolate_batch(gt_times, gt_positions, radar_times)
    interp_quats = rot_slerp_batch(gt_times, gt_quats, radar_times)
    poses = []
    for i, q_i in enumerate(interp_quats):
        quat_max = Rotation.from_quat(q_i).as_matrix()
        xyz = interp_positions[i].reshape(3,1)
        pose = np.hstack((quat_max, xyz))
        pose = np.vstack((pose, [0, 0, 0, 1]))
        poses.append(pose)
    return poses


def load_poses(poses_path):
    gt_times = []
    positions = []
    quats = []
    with open(poses_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            time, *data = np.fromstring(line, dtype=np.float64, sep=' ')
            gt_times.append(time)
            xyz = data[:3]
            quaternion = data[3:7]
            positions.append(xyz)
            quats.append(quaternion)
    return gt_times, positions, quats


def random_down_sample(pc, sample_points):
    submap_pcd = o3d.geometry.PointCloud()
    submap_pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    sampleA = random.sample(range(pc.shape[0]), sample_points)
    sampled_cloud = submap_pcd.select_by_index(sampleA)
    sampled_pc = np.array(sampled_cloud.points)
    return sampled_pc


def load_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        xyz = np.fromstring(lines[0], dtype=np.float32, sep=' ').reshape(3, 1)
        quaternion = np.fromstring(lines[1], dtype=np.float32, sep=' ')
        r = Rotation.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        sigle_T_Bs = np.hstack((rotation_matrix, xyz))  # 将xyz和rotation矩阵按列拼接
        sigle_T_Bs = np.vstack((sigle_T_Bs, [0, 0, 0, 1]))
        Bs_T_sigle = np.linalg.inv(sigle_T_Bs)
    return Bs_T_sigle


if __name__ == '__main__':
    data_path = '/path/to/radar_reloc_data/coloradar/'
    save_path = '/path/to/radar_reloc_data/'
    calib_path = data_path + 'calib/base_to_single_chip.txt'
    data_folder = 'rosbags'
    train_folder = 'train_short'
    test_folder = 'test_short'
    radartopic = '/mmWaveDataHdl/RScan'
    sub_size = 5
    target_points = 512
    seqs = ['edgar_classroom_run0', 'edgar_classroom_run1', 'edgar_classroom_run2', 'edgar_classroom_run4', 'edgar_classroom_run5',
            'ec_hallways_run0', 'ec_hallways_run1', 'ec_hallways_run2', 'ec_hallways_run3', 'ec_hallways_run4',
            'outdoors_run0', 'outdoors_run1', 'outdoors_run2', 'outdoors_run3', 'outdoors_run4']
    test_query_seqs = ['edgar_classroom_run5', 'ec_hallways_run4', 'outdoors_run4']

    Bs_T_sigle = load_calib(calib_path)

    for seq in tqdm(seqs):
        group_name = seq[:-5]
        seq_num = int(re.findall(r'\d+', seq)[-1])
        save_folder = train_folder if seq_num < 3 else test_folder
        gap_size = sub_size if seq in test_query_seqs else sub_size // 2
        
        seq_path = os.path.join(data_path, data_folder, seq + '.bag')
        gt_path = os.path.join(data_path, data_folder, seq + '_gt.txt')

        bag = rosbag.Bag(seq_path, "r")
        bag_data = bag.read_messages(radartopic)

        pointclouds = []
        radar_times = []
        gt_times, gt_positions, gt_quats = load_poses(gt_path)
        min_gt_time, max_gt_time = min(gt_times), max(gt_times)

        for topic, msg, t in bag_data:
            time = float('%6f' % (msg.header.stamp.to_sec()))
            if time < min_gt_time or time > max_gt_time:
                continue
            radar = pc2.read_points(msg, skip_nans=True, field_names=('x', 'y', 'z'))
            points = np.array(list(radar))
            points = np.column_stack((points, np.ones(points.shape[0])))
            pointclouds.append(points)
            radar_times.append(time)

        interp_poses = interpolation(np.array(gt_times), np.array(gt_positions),
                                     np.array(gt_quats), np.array(radar_times))

        # convert radar in global
        W_T_R = [pose.dot(Bs_T_sigle) for pose in interp_poses]

        submap_poses = []
        yaws = []
        count = 0
        for i in range(0, len(pointclouds), gap_size):
            end = i + sub_size
            if end >= len(pointclouds):
                continue
            submap_pc = np.empty((0,0), dtype=float, order='C')
            submap_pose = W_T_R[i+sub_size//2]
            for j in range(i, end):
                temp_pc = pointclouds[j]
                Rc_T_Rj = np.linalg.inv(submap_pose).dot(W_T_R[j])
                temp_pc_in_center = Rc_T_Rj.dot(temp_pc.T).T
                submap_pc = np.concatenate((submap_pc, temp_pc_in_center), axis=0) if submap_pc.size else temp_pc_in_center

            if len(submap_pc) < target_points:
                additional_points = target_points - len(submap_pc)
                sampled_points = submap_pc[np.random.choice(submap_pc.shape[0], additional_points), :]
                target_submap = np.concatenate((submap_pc, sampled_points), axis=0)[:, :3]
            else:
                target_submap = random_down_sample(submap_pc[:, :3], target_points)

            save_dir = os.path.join(save_path, save_folder, group_name, seq)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            submap_name = str(count).zfill(6) + ".bin"
            with open(os.path.join(save_dir, submap_name), 'wb') as f:
                target_submap.tofile(f)

            yaw = Rotation.from_matrix(submap_pose[:3,:3]).as_euler('xyz', degrees=True)[2]
            submap_poses.append(submap_pose)
            yaws.append(yaw)
            count += 1
            
        submaps_poses_path = os.path.join(save_path, save_folder, group_name, seq + '_poses.txt')
        with open(submaps_poses_path, 'w', encoding='utf-8') as f:
            for pose_id, pose in enumerate(submap_poses):
                pose_reshape = pose[:3, :4].reshape(1, 12).flatten()
                yaw_i = np.array([yaws[pose_id]])
                pose_with_yaw = np.concatenate((pose_reshape, yaw_i))
                f.write(' '.join(str(x) for x in pose_with_yaw) + '\n')