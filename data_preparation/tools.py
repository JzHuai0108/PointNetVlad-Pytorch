# -*-coding:utf-8-*-
import numpy as np
import random
from scipy.spatial.transform import Rotation
from test_slerp import pos_interpolate_batch, rot_slerp_batch

def interpolation(gt_times, gt_positions, gt_quats, radar_times):
    interp_positions = pos_interpolate_batch(gt_times, gt_positions, radar_times)
    interp_quats = rot_slerp_batch(gt_times, gt_quats, radar_times)
    poses = []
    for i, q_i in enumerate(interp_quats):
        quat_max = Rotation.from_quat(q_i).as_matrix()
        xyz = interp_positions[i].reshape(3, 1)
        pose = np.hstack((quat_max, xyz))
        pose = np.vstack((pose, [0, 0, 0, 1]))
        poses.append(pose)
    return poses

def random_down_sample(pc, sample_points):
    sampleA = random.sample(range(pc.shape[0]), sample_points)
    sampled_pc = pc[sampleA]
    return sampled_pc

def load_poses(poses_path, sign):
    positions = []
    quats = []
    gt_times = []
    with open(poses_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[1:]):
            time, *data = np.fromstring(line, dtype=np.float64, sep=sign)
            gt_times.append(time)
            xyz = data[:3]
            quaternion = data[3:7]
            positions.append(xyz)
            quats.append(quaternion)
    return gt_times, positions, quats
