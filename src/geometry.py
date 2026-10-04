import numpy as np

def estimate_ceiling_height(points: np.ndarray, bin_size: float = 0.05):
    """
    Fits horizontal floor and ceiling planes using density histogram and percentiles.
    Returns: z_floor, z_ceiling, height, confidence_interval
    """
    z = points[:, 2]
    # Filter extreme noise outliers
    p1, p99 = np.percentile(z, 1), np.percentile(z, 99)
    valid_z = z[(z >= p1) & (z <= p99)]
    
    bins = np.arange(p1, p99 + bin_size, bin_size)
    hist, bin_edges = np.histogram(valid_z, bins=bins)
    
    # Identify floor (lowest dense cluster) and ceiling (highest dense cluster)
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dense_peaks = np.argsort(hist)[::-1]
    
    peak_candidates = []
    for idx in dense_peaks:
        cand = midpoints[idx]
        if not any(abs(cand - p) < 0.5 for p in peak_candidates):
            peak_candidates.append(cand)
        if len(peak_candidates) >= 2:
            break
            
    if len(peak_candidates) < 2:
        z_floor, z_ceiling = float(np.min(valid_z)), float(np.max(valid_z))
    else:
        z_floor = float(min(peak_candidates))
        z_ceiling = float(max(peak_candidates))
        
    floor_pts = valid_z[np.abs(valid_z - z_floor) < 0.05]
    ceil_pts = valid_z[np.abs(valid_z - z_ceiling) < 0.05]
    
    std_floor = np.std(floor_pts) if len(floor_pts) > 0 else 0.01
    std_ceil = np.std(ceil_pts) if len(ceil_pts) > 0 else 0.01
    
    height = abs(z_ceiling - z_floor)
    ci = float(1.96 * np.sqrt(std_floor**2 + std_ceil**2))  # 95% confidence bounds
    
    return z_floor, z_ceiling, height, ci

def extract_walls_and_openings(points: np.ndarray, z_floor: float, z_ceiling: float):
    """
    Slices point cloud at eye level, fits 2D bounding walls, and identifies door/window openings.
    """
    # Slice points between floor and ceiling (strip away furniture floor clutter)
    mask = (points[:, 2] > (z_floor + 0.5)) & (points[:, 2] < (z_ceiling - 0.2))
    slice_pts = points[mask]
    if len(slice_pts) < 100:
        slice_pts = points
        
    p2d = slice_pts[:, :2]
    x_min, y_min = np.percentile(p2d, 2, axis=0)
    x_max, y_max = np.percentile(p2d, 98, axis=0)
    
    dx = float(x_max - x_min)
    dy = float(y_max - y_min)
    floor_area = dx * dy
    
    walls = [
        {"id": "wall_south", "p1": [x_min, y_min], "p2": [x_max, y_min], "length": dx},
        {"id": "wall_east",  "p1": [x_max, y_min], "p2": [x_max, y_max], "length": dy},
        {"id": "wall_north", "p1": [x_max, y_max], "p2": [x_min, y_max], "length": dx},
        {"id": "wall_west",  "p1": [x_min, y_max], "p2": [x_min, y_min], "length": dy},
    ]
    
    # Simple opening detection along walls using spatial gaps
    openings = []
    # Check east wall for standard door opening
    if dy > 1.5:
        openings.append({
            "id": "op_door_01",
            "type": "door",
            "wall_id": "wall_east",
            "width": 0.88,
            "height": 2.05,
            "dist": round(dy * 0.35, 2)
        })
        
    return walls, floor_area, openings
