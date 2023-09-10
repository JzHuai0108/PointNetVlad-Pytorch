# -*-coding:utf-8-*-
from tqdm import tqdm
import random
import os
import rosbag
import sensor_msgs.point_cloud2 as pc2
from test_slerp import *
import open3d as o3d

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


def load_inhouse_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()
        calib = np.fromstring(lines[0], dtype=np.float32, sep=' ').reshape(3, 4)
        bynav_T_r = np.vstack((calib, [0, 0, 0, 1]))
    return bynav_T_r

def comput_trans(I_T_R, c_pose, j_pose):
    N_R_Ic = c_pose[:3, :3]
    E_p_Ic = c_pose[:3, 3]
    N_R_Ij= j_pose[:3, :3]
    E_p_Ij = j_pose[:3, 3]
    Ic_R_Ij = N_R_Ic.T @ N_R_Ij
    Ic_p_Ij = N_R_Ic.T @ (E_p_Ij - E_p_Ic)
    trans = np.eye(4)
    trans[:3, :3] = Ic_R_Ij
    trans[:3, 3] = Ic_p_Ij
    Rc_T_Rj = np.linalg.inv(I_T_R) @ trans @ I_T_R
    return Rc_T_Rj

def comput_gt(center_pose, I_T_R):
    gt_pose = np.eye(4)
    N_R_R = center_pose[:3,:3] @ I_T_R[:3,:3]
    E_p_R = center_pose[:3,:3] @ I_T_R[:3, 3] + center_pose[:3, 3]
    gt_pose[:3, :3] = N_R_R
    gt_pose[:3, 3] = E_p_R
    return gt_pose

if __name__ == '__main__':
    data_path = '/path/to/radar_reloc_data/radar_reloc_data/inhouse/'
    save_path = '/path/to/radar_reloc_data/'
    calib_path = data_path + 'calib/bynav_T_radar.txt'
    train_folder = 'train_long'
    test_folder = 'test_long'
    radartopic = '/ars548'
    sub_size = 5
    target_points = 512
    data_folders = ['20221021', '20230814']
    groups_name = ['caochang', 'xinghu', 'xinbu']
    test_id = [7]

    I_T_R = load_inhouse_calib(calib_path)

    for folder in tqdm(data_folders):
        bag_files = [f for f in os.listdir(os.path.join(data_path, folder)) if f.endswith('.bag')]
        for i, bag in enumerate(bag_files):
            seq_name = bag.split('.')[0]
            seq_num = int(seq_name[-1])
            if seq_num == 1 or seq_num == 2:
                group_name = groups_name[0]
            elif seq_num == 3 or seq_num == 4:
                group_name = groups_name[1]
            else:
                group_name = groups_name[2]

            save_folder = train_folder if seq_num < 5 else test_folder
            gap_size = sub_size if seq_num in test_id else sub_size // 2

            seq_path = os.path.join(data_path, folder, bag)
            gt_path = os.path.join(data_path, folder, seq_name + '_bynavpose.txt')

            bag = rosbag.Bag(seq_path, "r")
            bag_data = bag.read_messages(radartopic)

            pointclouds = []
            radar_times = []
            gt_times, gt_positions, gt_quats = load_poses(gt_path)
            min_gt_time, max_gt_time = min(gt_times), max(gt_times)

            count = 0
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
            yaws = []
            count = 0
            for i in tqdm(range(0, len(pointclouds), gap_size)):
                end = i + sub_size
                if end >= len(pointclouds):
                    continue
                submap_pc = np.empty((0,0), dtype=float, order='C')
                center_pose = interp_poses[i+sub_size//2]
                for j in range(i, end):
                    temp_pc = pointclouds[j]
                    Rc_T_Rj = comput_trans(I_T_R, center_pose, interp_poses[j])
                    temp_pc_in_center = Rc_T_Rj.dot(temp_pc.T).T
                    submap_pc = np.concatenate((submap_pc, temp_pc_in_center), axis=0) if submap_pc.size else temp_pc_in_center

                submap_pc = submap_pc[np.linalg.norm(submap_pc[:, :3], axis=1) <= 200]


                if len(submap_pc) < target_points:
                    additional_points = target_points - len(submap_pc)
                    sampled_points = submap_pc[np.random.choice(submap_pc.shape[0], additional_points), :]
                    target_submap = np.concatenate((submap_pc, sampled_points), axis=0)[:, :3]
                else:
                    target_submap = random_down_sample(submap_pc[:, :3], target_points)

                save_dir = os.path.join(save_path, save_folder, group_name, seq_name)
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, (str(count).zfill(6) + ".bin")), 'wb') as f:
                    target_submap.tofile(f)

                submap_pose = comput_gt(center_pose, I_T_R)
                yaw = Rotation.from_matrix(submap_pose[:3, :3]).as_euler('xyz', degrees=True)[2]
                yaws.append(yaw)
                submap_poses.append(submap_pose)
                count += 1

            submaps_poses_path = os.path.join(save_path, save_folder, group_name, seq_name + '_poses.txt')
            with open(submaps_poses_path, 'w', encoding='utf-8') as f:
                for pose_id, pose in enumerate(submap_poses):
                    pose_reshape = pose[:3, :4].reshape(1, 12).flatten()
                    yaw_i = np.array([yaws[pose_id]])
                    pose_with_yaw = np.concatenate((pose_reshape, yaw_i))
                    f.write(' '.join(str(x) for x in pose_with_yaw) + '\n')
