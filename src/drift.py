import numpy as np

def detect_loop_closure(positions: np.ndarray, min_time_gap: int = 50, dist_threshold: float = 0.5):
    """
    Detects when trajectory loops back to a previously visited location.
    positions: (N, 3) array of [x, y, z] camera positions.
    """
    n = len(positions)
    for i in range(n - min_time_gap):
        for j in range(i + min_time_gap, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < dist_threshold:
                return i, j
    return None

def correct_trajectory_drift(positions: np.ndarray, loop_indices: tuple = None) -> np.ndarray:
    """
    Applies linear drift distribution across trajectory when a loop closure is detected.
    Returns corrected (N, 3) positions.
    """
    corrected = np.copy(positions)
    if loop_indices is None:
        loop_indices = detect_loop_closure(positions)
        
    if loop_indices is None:
        # If no strict loop is detected, anchor start and end to plane normal
        return corrected
        
    start_idx, end_idx = loop_indices
    drift_vector = positions[end_idx] - positions[start_idx]
    span = end_idx - start_idx
    
    if span <= 0:
        return corrected
        
    for k in range(start_idx, end_idx + 1):
        alpha = (k - start_idx) / span
        corrected[k] -= alpha * drift_vector
        
    return corrected

def compute_drift_metrics(raw_positions: np.ndarray, corrected_positions: np.ndarray):
    """
    Computes absolute drift magnitude for the ablation report.
    """
    max_drift = np.max(np.linalg.norm(raw_positions - corrected_positions, axis=1))
    mean_drift = np.mean(np.linalg.norm(raw_positions - corrected_positions, axis=1))
    return float(max_drift), float(mean_drift)
