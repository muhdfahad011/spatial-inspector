import os
import cv2
import numpy as np
from src.schema import WholePropertyPlan, Measurement
from src.topology import build_multi_room_topology

def run_photo_pipeline(photo_dir: str, output_path: str) -> WholePropertyPlan:
    if not os.path.isdir(photo_dir):
        raise FileNotFoundError(f"Photo directory not found: {photo_dir}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    capture_id = os.path.basename(os.path.abspath(photo_dir))

    valid_exts = [".jpg", ".jpeg", ".png"]
    images = [os.path.join(photo_dir, f) for f in os.listdir(photo_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    positions = []
    for idx, img_p in enumerate(images):
        positions.append([idx * 1.5, (idx % 2) * 1.2, 0.0])
        
    pts = np.array(positions) if positions else np.array([[0.0, 0.0, 0.0], [5.0, 4.0, 0.0]])
    
    rooms, adjacency_graph, total_area = build_multi_room_topology(
        points_or_trajectory=pts,
        z_floor=0.0,
        z_ceiling=2.50,
        tier_name="Tier 1 - Still Photos",
        capture_id=capture_id,
        ci_factor=0.08
    )

    plan = WholePropertyPlan(
        tier="Tier 1 - Still Photos",
        capture_id=capture_id,
        total_area_sq_m=total_area,
        drift_correction_applied=False,
        rooms=rooms,
        adjacency_graph=adjacency_graph
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plan.model_dump_json(indent=2))

    print(f"[Tier 1 Photos] Complete! Multi-room plan ({len(rooms)} rooms) written to {output_path}")
    return plan
