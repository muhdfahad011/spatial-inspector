import os
import cv2
import numpy as np
from src.schema import WholePropertyPlan, Measurement
from src.topology import build_multi_room_topology

def run_video_pipeline(video_path: str, output_path: str) -> WholePropertyPlan:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_stem = os.path.splitext(os.path.basename(video_path))[0]
    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(video_path)))
    capture_id = parent_dir if file_stem.lower() in ["rgb", "video", "walkthrough"] and parent_dir else file_stem

    print(f"[Tier 2 Video] Ingesting walkthrough clip from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    trajectory = []
    curr_pos = np.array([0.0, 0.0, 0.0])
    trajectory.append(curr_pos.copy())

    prev_gray = None
    frame_idx = 0
    max_frames_to_process = 60
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or processed_count >= max_frames_to_process:
            break
        frame_idx += 1
        if frame_idx % 6 != 0:
            continue

        h, w = frame.shape[:2]
        if w > 320:
            scale = 320.0 / w
            frame = cv2.resize(frame, (320, int(h * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            dx = float(np.mean(flow[..., 0])) * 0.03
            dy = float(np.mean(flow[..., 1])) * 0.03
            curr_pos[0] += dx
            curr_pos[1] += dy
            trajectory.append(curr_pos.copy())
            processed_count += 1
        prev_gray = gray
    cap.release()

    traj_arr = np.array(trajectory)
    z_floor = 0.0
    z_ceiling = 2.45

    rooms, adjacency_graph, total_area = build_multi_room_topology(
        points_or_trajectory=traj_arr,
        z_floor=z_floor,
        z_ceiling=z_ceiling,
        tier_name="Tier 2 - Video Walkthrough",
        capture_id=capture_id,
        ci_factor=0.03
    )

    plan = WholePropertyPlan(
        tier="Tier 2 - Video Walkthrough",
        capture_id=capture_id,
        total_area_sq_m=total_area,
        drift_correction_applied=False,
        rooms=rooms,
        adjacency_graph=adjacency_graph
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plan.model_dump_json(indent=2))

    print(f"[Tier 2 Video] Optical flow tracking complete. Multi-room plan generated: {len(rooms)} rooms.")
    print(f"[Tier 2 Video] Complete! Output written to {output_path}")
    return plan
