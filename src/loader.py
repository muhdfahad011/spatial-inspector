import os
import glob
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R

def load_camera_matrix(csv_path: str) -> np.ndarray:
    """Loads 3x3 camera intrinsic matrix from CSV."""
    return np.loadtxt(csv_path, delimiter=",")

def load_odometry(csv_path: str) -> pd.DataFrame:
    """Loads camera trajectory poses and intrinsics per frame."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    return df

def depth_to_point_cloud(depth_img: np.ndarray, K: np.ndarray, depth_scale: float = 1000.0) -> np.ndarray:
    """
    Unprojects 2D depth map to 3D camera coordinates.
    depth_scale converts depth units to meters (mm -> m).
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img.astype(np.float32) / depth_scale
    
    valid = (z > 0.1) & (z < 6.0)
    u_val = u[valid]
    v_val = v[valid]
    z_val = z[valid]
    
    # Scale intrinsics to depth image resolution
    fx = K[0, 0] * (w / 1920.0)
    fy = K[1, 1] * (h / 1440.0)
    cx = K[0, 2] * (w / 1920.0)
    cy = K[1, 2] * (h / 1440.0)
    
    x_val = (u_val - cx) * z_val / fx
    y_val = (v_val - cy) * z_val / fy
    
    pts_cam = np.stack([x_val, y_val, z_val], axis=-1)
    return pts_cam

def transform_points(pts: np.ndarray, position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Transforms 3D points from camera frame to world frame."""
    # quat is [qx, qy, qz, qw]
    rot = R.from_quat(quat).as_matrix()
    pts_world = (rot @ pts.T).T + position
    return pts_world

def load_capture_point_cloud(scan_dir: str, step: int = 15, max_points: int = 250000) -> np.ndarray:
    """
    Aggregates point cloud from a capture scan directory across keyframes.
    """
    cam_csv = os.path.join(scan_dir, "camera_matrix.csv")
    odo_csv = os.path.join(scan_dir, "odometry.csv")
    depth_dir = os.path.join(scan_dir, "depth")
    
    K = load_camera_matrix(cam_csv)
    odo_df = load_odometry(odo_csv)
    
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    sampled_files = depth_files[::step]
    
    world_points = []
    
    for df_path in sampled_files:
        frame_id = int(os.path.splitext(os.path.basename(df_path))[0])
        row = odo_df[odo_df["frame"] == frame_id]
        if row.empty:
            continue
            
        row = row.iloc[0]
        pos = np.array([row["x"], row["y"], row["z"]], dtype=np.float32)
        quat = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float32)
        
        depth_img = cv2.imread(df_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            continue
            
        pts_cam = depth_to_point_cloud(depth_img, K)
        pts_w = transform_points(pts_cam, pos, quat)
        
        # Subsample to keep memory and computation clean
        if len(pts_w) > 2000:
            idx = np.random.choice(len(pts_w), 2000, replace=False)
            pts_w = pts_w[idx]
            
        world_points.append(pts_w)
        
    merged = np.vstack(world_points)
    if len(merged) > max_points:
        idx = np.random.choice(len(merged), max_points, replace=False)
        merged = merged[idx]
        
    return merged
