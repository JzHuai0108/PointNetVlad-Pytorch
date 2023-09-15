# -*-coding:utf-8-*-
import numpy as np
import os
import rosbag
import argparse
from tqdm import tqdm
import sensor_msgs.point_cloud2 as pc2
from tools import interpolation, random_down_sample, load_poses
from scipy.spatial.transform import Rotation

def load_inhouse_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        calib = np.fromstring(lines[0], dtype=np.float64, sep=' ').reshape(3, 4)
        bynav_T_r = np.vstack((calib, [0, 0, 0, 1]))
    return bynav_T_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/path/to/radar_reloc_data/', help='radar datasets path')
    parser.add_argument('--dataset_name', type=str, default='inhouse', help='radar dataset folder')
    parser.add_argument('--data_folders', type=list, default=['20221021', '20230814'], help='radar data folder')
    parser.add_argument('--groups_name', type=list, default=['caochang', 'xinghu', 'xinbu'], help='the groups name in the dataset')
    parser.add_argument('--train_folder', type=str, default='train_long', help='train submaps saved folder')
    parser.add_argument('--test_folder', type=str, default='test_long', help='test submaps saved folder')
    parser.add_argument('--radar_topic', type=str, default='/ars548', help='radar_topic in rosbag')
    parser.add_argument('--frame_winsize', type=int, default=5, help='window size for submap')
    parser.add_argument('--test_seqs', type=list, default=[7], help='test seqs in the dataset')
    parser.add_argument('--target_points', type=int, default=512, help='traget points in saved submaps')
    cfgs = parser.parse_args()

    dataset_path = os.path.join(cfgs.data_path, cfgs.dataset_name)
    calib_path = os.path.join(dataset_path, 'calib/bynav_T_radar.txt')

    I_T_R = load_inhouse_calib(calib_path)

    for folder in tqdm(cfgs.data_folders):
        bag_files = [f for f in os.listdir(os.path.join(dataset_path, folder)) if f.endswith('.bag')]
        for i, bag in enumerate(bag_files):
            seq_name = bag.split('.')[0]
            seq_num = int(seq_name[-1])
            if seq_num == 1 or seq_num == 2:
                group_name = cfgs.groups_name[0]
            elif seq_num == 3 or seq_num == 4:
                group_name = cfgs.groups_name[1]
            else:
                group_name = cfgs.groups_name[2]

            save_folder = cfgs.train_folder if seq_num < 5 else cfgs.test_folder
            save_dir = os.path.join(dataset_path, save_folder, group_name, seq_name)
            os.makedirs(save_dir, exist_ok=True)

            gap_size = cfgs.frame_winsize if seq_num in cfgs.test_seqs else cfgs.frame_winsize // 2

            seq_path = os.path.join(dataset_path, folder, bag)
            gt_path = os.path.join(dataset_path, folder, seq_name + '_bynavpose.txt')

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

                if len(submap_pc) < cfgs.target_points:
                    additional_points = cfgs.target_points - len(submap_pc)
                    sampled_points = submap_pc[np.random.choice(submap_pc.shape[0], additional_points), :]
                    target_submap = np.concatenate((submap_pc, sampled_points), axis=0)[:, :3]
                else:
                    target_submap = random_down_sample(submap_pc[:, :3], cfgs.target_points)

                submap_name = str(submap_time) + ".bin"
                with open(os.path.join(save_dir, submap_name), 'wb') as f:
                    target_submap.tofile(f)

                submap_pose = np.matmul(center_pose, I_T_R)
                submap_poses.append(submap_pose)
                yaw = Rotation.from_matrix(submap_pose[:3, :3]).as_euler('xyz', degrees=True)[2]
                yaws.append(yaw)
                submap_timestamps.append(submap_time)

            submaps_poses_path = save_dir + '_poses.txt'
            with open(submaps_poses_path, 'w', encoding='utf-8') as f:
                for pose_id, pose in enumerate(submap_poses):
                    pose_reshape = pose[:3, :4].reshape(1, 12).flatten()
                    time_i = [str(submap_timestamps[pose_id])]
                    yaw_i = np.array([yaws[pose_id]])
                    pose_with_yaw = np.concatenate((time_i, pose_reshape, yaw_i))
                    f.write(' '.join(str(x) for x in pose_with_yaw) + '\n')
