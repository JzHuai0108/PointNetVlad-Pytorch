# -*-coding:utf-8-*-
import numpy as np
from tqdm import tqdm
import os
import rosbag
import argparse
import sensor_msgs.point_cloud2 as pc2
from tools import interpolation, random_down_sample, load_poses
from scipy.spatial.transform import Rotation

def load_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        xyz = np.fromstring(lines[0], dtype=np.float64, sep=' ').reshape(3, 1)
        quaternion = np.fromstring(lines[1], dtype=np.float64, sep=' ')
        r = Rotation.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        Bs_T_singlechip = np.hstack((rotation_matrix, xyz))
        Bs_T_singlechip = np.vstack((Bs_T_singlechip, [0, 0, 0, 1]))
    return Bs_T_singlechip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/path/to/radar_reloc_data/',
                        help='radar datasets path')
    parser.add_argument('--dataset_name', type=str, default='coloradar', help='radar dataset folder')
    parser.add_argument('--data_folder', type=str, default='rosbags', help='radar data folder')
    parser.add_argument('--seqs', type=list,
                        default=['edgar_classroom_run1', 'edgar_classroom_run2', 'edgar_classroom_run4', 'edgar_classroom_run5',
                                 'arpg_lab_run0', 'arpg_lab_run1', 'arpg_lab_run2', 'arpg_lab_run3', 'arpg_lab_run4',
                                 'outdoors_run0', 'outdoors_run1', 'outdoors_run2', 'outdoors_run3', 'outdoors_run4',
                                 'edgar_army_run0', 'edgar_army_run1', 'edgar_army_run2', 'edgar_army_run3', 'edgar_army_run4', 'edgar_army_run5'],
                        help='the groups name in the dataset')
    parser.add_argument('--test_database_seqs', type=list, default=['edgar_classroom_run4', 'arpg_lab_run3', 'outdoors_run3', 'edgar_army_run4'],
                        help='the groups name in database_seqs')
    parser.add_argument('--test_query_seqs', type=list, default=['edgar_classroom_run5', 'arpg_lab_run4', 'outdoors_run4', 'edgar_army_run5'],
                        help='the groups name in query_seqs')
    parser.add_argument('--train_folder', type=str, default='train_short', help='train submaps saved folder')
    parser.add_argument('--test_folder', type=str, default='test_short', help='test submaps saved folder')
    parser.add_argument('--radar_topic', type=str, default='/mmWaveDataHdl/RScan', help='radar_topic in rosbag')
    parser.add_argument('--frame_winsize', type=int, default=5, help='window size for submap')
    parser.add_argument('--target_points', type=int, default=512, help='traget points in saved submaps')
    cfgs = parser.parse_args()

    dataset_path = os.path.join(cfgs.data_path, cfgs.dataset_name)
    calib_path = os.path.join(dataset_path, 'calib/base_to_single_chip.txt')


    Bs_T_singlechip = load_calib(calib_path)

    for seq in tqdm(cfgs.seqs):
        group_name = seq[:-5]
        save_folder = cfgs.train_folder if seq not in cfgs.test_database_seqs and seq not in cfgs.test_query_seqs else cfgs.test_folder
        save_dir = os.path.join(cfgs.data_path, save_folder, group_name, seq)
        os.makedirs(save_dir, exist_ok=True)

        gap_size = cfgs.frame_winsize if seq in cfgs.test_query_seqs else cfgs.frame_winsize // 2
        
        seq_path = os.path.join(dataset_path, cfgs.data_folder, seq + '.bag')
        gt_path = os.path.join(dataset_path, cfgs.data_folder, seq + '_gt.txt')

        bag = rosbag.Bag(seq_path, "r")
        bag_data = bag.read_messages(cfgs.radar_topic)

        pointclouds = []
        radar_times = []
        gt_times, gt_positions, gt_quats = load_poses(gt_path, ' ')
        min_gt_time, max_gt_time = min(gt_times), max(gt_times)

        for topic, msg, t in bag_data:
            time = float('%6f' % (msg.header.stamp.to_sec()))
            if time < min_gt_time or time > max_gt_time:  # So the interpolation for poses will succeed.
                continue
            radar = pc2.read_points(msg, skip_nans=True, field_names=('x', 'y', 'z'))
            points = np.array(list(radar))
            points = np.column_stack((points, np.ones(points.shape[0])))
            pointclouds.append(points)
            radar_times.append(time)

        interp_poses = interpolation(np.array(gt_times), np.array(gt_positions),
                                     np.array(gt_quats), np.array(radar_times))

        W_T_R = [np.matmul(pose, Bs_T_singlechip) for pose in interp_poses]

        submap_poses = []
        submap_timestamps = []
        yaws = []
        for i in range(0, len(pointclouds), gap_size):
            end = i + cfgs.frame_winsize
            if end >= len(pointclouds):
                continue
            submap_pc = np.empty((0,0), dtype=float, order='C')
            submap_time = int(radar_times[i + cfgs.frame_winsize // 2] * 1e6)
            submap_pose = W_T_R[i+cfgs.frame_winsize//2]
            for j in range(i, end):
                temp_pc = pointclouds[j]
                Rc_T_Rj = np.matmul(np.linalg.inv(submap_pose), W_T_R[j])
                temp_pc_in_center = np.matmul(Rc_T_Rj, temp_pc.T).T
                submap_pc = np.concatenate((submap_pc, temp_pc_in_center), axis=0) if submap_pc.size else temp_pc_in_center

            if len(submap_pc) < cfgs.target_points:
                additional_points = cfgs.target_points - len(submap_pc)
                sampled_points = submap_pc[np.random.choice(submap_pc.shape[0], additional_points), :]
                target_submap = np.concatenate((submap_pc, sampled_points), axis=0)[:, :3]
            else:
                target_submap = random_down_sample(submap_pc[:, :3], cfgs.target_points)

            submap_name = str(submap_time) + ".bin"
            with open(os.path.join(save_dir, submap_name), 'wb') as f:
                target_submap.tofile(f)

            yaw = Rotation.from_matrix(submap_pose[:3, :3]).as_euler('xyz', degrees=True)[2]
            yaws.append(yaw)
            submap_timestamps.append(submap_time)
            submap_poses.append(submap_pose)
            
        submaps_poses_path = save_dir + '_poses.txt'
        with open(submaps_poses_path, 'w', encoding='utf-8') as f:
            for pose_id, pose in enumerate(submap_poses):
                pose_reshape = pose[:3, :4].reshape(1, 12).flatten()
                time_i = [str(submap_timestamps[pose_id])]
                yaw_i = np.array([yaws[pose_id]])
                pose_with_yaw = np.concatenate((time_i, pose_reshape, yaw_i))
                f.write(' '.join(str(x) for x in pose_with_yaw) + '\n')