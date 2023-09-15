# -*-coding:utf-8-*-
import numpy as np
from tqdm import tqdm
import os
import rosbag
import sensor_msgs.point_cloud2 as pc2
from tools import interpolation, random_down_sample, load_poses
from scipy.spatial.transform import Rotation
import argparse

def load_mscrad4r_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        calib = np.fromstring(lines[0], dtype=np.float64, sep=' ').reshape(3, 4)
        imu_T_radar = np.vstack((calib, [0, 0, 0, 1]))
    return imu_T_radar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/path/to/radar_reloc_data/', help='radar datasets path')
    parser.add_argument('--dataset_name', type=str, default='mscrad4r', help='radar dataset folder')
    parser.add_argument('--seqs', type=list, default=['RURAL_C0', 'RURAL_C1', 'RURAL_D0', 'RURAL_D1',
                                                      'RURAL_E0', 'RURAL_E1'], help='the groups name in the dataset')
    parser.add_argument('--train_folder', type=str, default='train_long', help='train submaps saved folder')
    parser.add_argument('--test_folder', type=str, default='test_long', help='test submaps saved folder')
    parser.add_argument('--radar_topic', type=str, default='/oculii_radar/point_cloud', help='radar_topic in rosbag')
    parser.add_argument('--frame_winsize', type=int, default=3, help='window size for submap')
    parser.add_argument('--target_points', type=int, default=512, help='traget points in saved submaps')
    cfgs = parser.parse_args()

    dataset_path = os.path.join(cfgs.data_path, cfgs.dataset_name)
    calib_path = os.path.join(dataset_path, 'imu_T_radar.txt')

    I_T_R = load_mscrad4r_calib(calib_path)

    for seq in tqdm(cfgs.seqs):
        group_name = seq[:-1]
        tra_id = seq.split('_')[-1]
        if 'E' not in tra_id:
            save_folder = cfgs.train_folder
            gap_size = cfgs.frame_winsize // 2
        else:
            seq_num = int(tra_id[-1])
            save_folder = cfgs.test_folder
            gap_size = cfgs.frame_winsize // 2 if seq_num == 0 else cfgs.frame_winsize

        save_dir = os.path.join(cfgs.data_path, save_folder, group_name, seq)
        os.makedirs(save_dir, exist_ok=True)

        seq_path = os.path.join(dataset_path, seq, seq + '.bag')
        gt_path = os.path.join(dataset_path, seq, seq + '_gt.txt')

        bag = rosbag.Bag(seq_path, "r")
        bag_data = bag.read_messages(cfgs.radar_topic)

        pointclouds = []
        radar_times = []
        gt_times, gt_positions, gt_quats = load_poses(gt_path, ',')
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

        submap_poses = []
        submap_timestamps = []
        yaws = []
        for i in tqdm(range(0, len(pointclouds), gap_size)):
            end = i + cfgs.frame_winsize
            if end >= len(pointclouds):
                continue
            submap_pc = np.empty((0,0), dtype=float, order='C')
            submap_time = int(radar_times[i + cfgs.frame_winsize // 2] * 1e6)
            center_pose = interp_poses[i + cfgs.frame_winsize // 2]

            for j in range(i, end):
                temp_pc = pointclouds[j]
                Rc_T_E = np.linalg.inv(center_pose @ I_T_R)
                E_T_Rj = interp_poses[j] @ I_T_R
                Rc_T_Rj = np.matmul(Rc_T_E, E_T_Rj)
                temp_pc_in_center = np.matmul(Rc_T_Rj, temp_pc.T).T
                submap_pc = np.concatenate((submap_pc, temp_pc_in_center), axis=0) if submap_pc.size else temp_pc_in_center

            submap_pc = submap_pc[np.linalg.norm(submap_pc[:, :3], axis=1) <= 200]

            target_submap = random_down_sample(submap_pc[:, :3], cfgs.target_points)
            ones = np.ones((target_submap.shape[0], 1))
            target_submap = np.hstack((target_submap, ones))
            target_submap = np.matmul(I_T_R, target_submap.T).T[:, :3]

            submap_name = str(submap_time) + ".bin"
            with open(os.path.join(save_dir, submap_name), 'wb') as f:
                target_submap.tofile(f)

            submap_pose = center_pose
            submap_poses.append(submap_pose)
            yaw = Rotation.from_matrix(submap_pose[:3, :3]).as_euler('xyz', degrees=True)[2]
            yaws.append(yaw)
            submap_timestamps.append(submap_time)

        submaps_poses_path = os.path.join(cfgs.data_path, save_folder, group_name, seq + '_poses.txt')
        with open(submaps_poses_path, 'w', encoding='utf-8') as f:
            for pose_id, pose in enumerate(submap_poses):
                pose_reshape = pose[:3, :4].reshape(1, 12).flatten()
                time_i = [str(submap_timestamps[pose_id])]
                yaw_i = np.array([yaws[pose_id]])
                pose_with_yaw = np.concatenate((time_i, pose_reshape, yaw_i))
                f.write(' '.join(str(x) for x in pose_with_yaw) + '\n')
