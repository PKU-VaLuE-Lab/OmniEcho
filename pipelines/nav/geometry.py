#!/usr/bin/env python3

"""OmniEcho processing component."""
from __future__ import annotations

import os
from settings import DATA_ROOT as TRAJECTORY_ROOT, NAV_ROOT as OUTPUT_ROOT, MODEL_ROOT, API_BASE_URL as SERVICE_URL, API_MODEL

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import wavfile

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
HEIGHT = 125
PITCH_1 = 0      
PITCH_2 = 30     
SAMPLE_STEP = 4
NUM_HISTORY = 8
NUM_FUTURE_STEPS = 4
PREDICT_STEP_NUM = 32
PIXEL_GOAL_ONLY = True

MIN_DIST = 0.3   


# =============================================================================

# =============================================================================
def get_trajectory_relative_to_frame(extrinsics, camera_deg=0):
    """OmniEcho processing component."""
    T_camera2robot = np.array(
        [[[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    T_robot2camera = np.array(
        [[[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    if camera_deg is not None:
        camera_rad = np.radians(camera_deg)
        T_deg = np.array(
            [[[1.0, 0.0, 0.0, 0.0],
              [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
              [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
              [0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
        T_robot2camera = np.matmul(T_robot2camera, T_deg)
        T_camera2robot = np.linalg.inv(T_robot2camera)
    extrinsics_robot = np.matmul(extrinsics, T_camera2robot)
    T_ref = extrinsics_robot[0]
    T_ref_inv = np.linalg.inv(T_ref)
    relative_to_ref = np.matmul(T_ref_inv[np.newaxis, :, :], extrinsics_robot)
    relative_translations = relative_to_ref[:, :2, 3]
    relative_yaws = np.arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])
    return np.concatenate((relative_translations, relative_yaws.reshape(-1, 1)), axis=-1)


def smooth_and_resample_trajectory(points, sample_length=33, interval=0.1):
    total_distance = sample_length * interval
    if len(points) == 0:
        return np.zeros((sample_length, 2))
    if len(points) == 1:
        return np.tile(points[0], (sample_length, 1))
    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)
    if len(points) > 3:
        cs_x = CubicSpline(cumulative_distances, points[:, 0])
        cs_y = CubicSpline(cumulative_distances, points[:, 1])
        dense_distances = np.linspace(0, cumulative_distances[-1], max(50, len(points) * 2))
        smoothed_points = np.column_stack((cs_x(dense_distances), cs_y(dense_distances)))
        smooth_diff = np.diff(smoothed_points, axis=0)
        smooth_seg = np.sqrt(np.sum(smooth_diff ** 2, axis=1))
        smooth_cumulative_distances = np.insert(np.cumsum(smooth_seg), 0, 0)
    else:
        smoothed_points = points
        smooth_cumulative_distances = cumulative_distances
    target_distances = np.linspace(0, total_distance, sample_length)
    resampled = np.zeros((sample_length, 2))
    for i, target_dist in enumerate(target_distances):
        if target_dist >= smooth_cumulative_distances[-1]:
            resampled[i] = smoothed_points[-1]
            continue
        seg = np.searchsorted(smooth_cumulative_distances, target_dist, side='right') - 1
        start_dist = smooth_cumulative_distances[seg]
        end_dist = smooth_cumulative_distances[seg + 1]
        t = (target_dist - start_dist) / (end_dist - start_dist)
        resampled[i] = smoothed_points[seg] + t * (smoothed_points[seg + 1] - smoothed_points[seg])
    return resampled


def xy_to_delta_xyt(xy_actions):
    vectors = np.diff(xy_actions, axis=0)
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])
    delta_yaw = np.diff(yaw)
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi
    delta_yaw = np.concatenate([[yaw[0]], delta_yaw])
    return np.concatenate([vectors, delta_yaw[:, None]], axis=1)


def interpolate_and_resample_trajectory(absolute_trajectories, predict_step_num=None):
    start_point = np.array([[0.0, 0.0]])
    traj = absolute_trajectories[..., :2]
    steps = traj[1:] - traj[:-1]
    mask = (steps ** 2).sum(axis=-1) > 0.05
    filtered_traj = np.concatenate([start_point, traj[1:][mask]], axis=0)
    resampled_trajectories = smooth_and_resample_trajectory(filtered_traj, sample_length=predict_step_num + 1)
    resampled_relative_poses = xy_to_delta_xyt(resampled_trajectories)
    resampled_relative_poses[:, 0:2] *= 4
    return resampled_trajectories, resampled_relative_poses


# =============================================================================

# =============================================================================
def _mat(x):
    """OmniEcho processing component."""
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def load_episode(scene_dir, episode_id, height=HEIGHT, pitch_1=PITCH_1, pitch_2=PITCH_2):
    import pyarrow.parquet as pq

    """OmniEcho processing component."""
    pq_path = os.path.join(scene_dir, "data", f"chunk-{episode_id // 1000:03d}",
                           f"episode_{episode_id:06d}.parquet")
    df = pq.read_table(pq_path).to_pandas()

    actions = df["action"].tolist()
    c2w_1 = np.stack([_mat(x) for x in df[f"pose.{height}cm_{pitch_1}deg"].tolist()])   
    c2w_2 = np.stack([_mat(x) for x in df[f"pose.{height}cm_{pitch_2}deg"].tolist()])   

    rgf_key = f"relative_goal_frame_id.{height}cm_{pitch_2}deg"
    goal_key = f"goal.{height}cm_{pitch_2}deg"
    pixel_goals = [[int(np.asarray(df[rgf_key][i])), np.asarray(df[goal_key][i]).tolist()]
                   for i in range(len(df))]
    return actions, c2w_1, c2w_2, pixel_goals


def enumerate_pixel_goal_samples(actions, pixel_goals, sample_step=SAMPLE_STEP):
    """OmniEcho processing component."""
    actions = actions[1:] + [0]
    actions_len = len(actions)
    samples = []
    if actions_len < 4:
        return samples
    num_rounds = actions_len // sample_step
    for n in range(num_rounds + 1):
        if n * sample_step == actions_len or n * sample_step == actions_len - 1:
            continue
        start = n * sample_step
        pg = pixel_goals[start]
        if pg[0] == -1:
            continue  
        goal_len = pg[0]
        if goal_len < 3:
            continue
        samples.append((start, goal_len, pg[1]))
    return samples


# =============================================================================

# =============================================================================
def load_mono(path):
    """OmniEcho processing component."""
    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)       
    return sr, data


def source_dir_local(c2w_listener, source_world):
    """OmniEcho processing component."""
    R_c2w = c2w_listener[:3, :3]
    C = c2w_listener[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ C
    p_cam = R_w2c @ source_world + t_w2c   
    front, left, up = p_cam[2], -p_cam[0], -p_cam[1]
    v = np.array([front, left, up], dtype=np.float64)
    d = float(np.linalg.norm(v))
    unit = v / d if d > 1e-9 else np.array([1.0, 0.0, 0.0])
    return unit, d


def encode_static_foa(mono, unit, dist, min_dist=MIN_DIST):
    """OmniEcho processing component."""
    d = max(dist, min_dist)
    sig = mono * min_dist / d
    W = sig
    Y = sig * unit[1]   
    Z = sig * unit[2]   
    X = sig * unit[0]   
    return np.stack([W, Y, Z, X], axis=0)   # [4, n]


def write_foa_wav(path, foa, sr):
    """OmniEcho processing component."""
    wavfile.write(path, sr, foa.T.astype(np.float32))


# =============================================================================

# =============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="geometry.py self-check: read one OmniEchoBench-Nav episode and print geometry")
    ap.add_argument("--data_root",
                    default=TRAJECTORY_ROOT)
    ap.add_argument("--dataset", default="OmniEchoBench-Nav")
    ap.add_argument("--scene", default="00000-kfPV7w3FaU5")
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    scene_dir = os.path.join(args.data_root, args.dataset, args.scene)
    actions, c2w_1, c2w_2, pixel_goals = load_episode(scene_dir, args.episode)
    source_world = c2w_1[-1, :3, 3].copy()
    print(f"[selftest] scene={args.scene} ep={args.episode} frames={len(c2w_1)}")
    print(f"[selftest] frame0 center (world, z=height): {np.round(c2w_1[0, :3, 3], 3)}")
    print(f"[selftest] source_world (goal center):      {np.round(source_world, 3)}")

    print("[selftest] Listener-to-source direction (distance should approach zero as the frame index increases):")
    prev_d = None
    mono_ok = True
    for fr in range(0, len(c2w_1), max(1, len(c2w_1) // 6)):
        unit, d = source_dir_local(c2w_1[fr], source_world)
        az = float(np.degrees(np.arctan2(unit[1], unit[0])))
        el = float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))
        print(f"           frame {fr:3d}: dist={d:5.2f}m  az={az:+7.1f}(positive=left)  el={el:+6.1f}")
        if prev_d is not None and d > prev_d + 1.5:
            mono_ok = False
        prev_d = d

    samples = enumerate_pixel_goal_samples(actions, pixel_goals)
    print(f"[selftest] pixel_goal samples: {len(samples)}")
    if samples:
        start, goal_len, pg = samples[0]
        traj = get_trajectory_relative_to_frame(c2w_2[start:start + goal_len + 1], camera_deg=PITCH_2)
        print(f"[selftest] Sample 0: start={start} goal_len={goal_len} pixel_goal={pg}")
        print(f"[selftest] First 3 trajectory steps (x=forward, y=left, yaw): {np.round(traj[1:4], 3).tolist()}")
        print(f"[selftest] Net displacement at trajectory endpoint: {np.round(traj[-1], 3).tolist()} (forward x should be > 0)")
    print("[selftest] Distance generally decreases toward the endpoint:", "OK" if mono_ok else "NOTE: OmniEchoBench-Nav paths may curve, allowing local distance increases")
