import os
import argparse
import numpy as np
from src.schema import WholePropertyPlan, Measurement
from src.loader import load_capture_point_cloud
from src.geometry import estimate_ceiling_height
from src.drift import assess_and_correct_drift
from src.video_pipeline import run_video_pipeline
from src.photo_pipeline import run_photo_pipeline
from src.topology import build_multi_room_topology

def run_reconstruction(scan_path: str, output_path: str) -> WholePropertyPlan:
    if os.path.isfile(scan_path) and scan_path.lower().endswith(".mp4"):
        return run_video_pipeline(scan_path, output_path)

    if os.path.isdir(scan_path):
        has_ply = any(f.endswith(".ply") for f in os.listdir(scan_path))
        has_depth = any(f.endswith((".npz", ".npy", ".bin")) or "depth" in f.lower() for f in os.listdir(scan_path))
        if not has_ply and not has_depth:
            has_images = any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(scan_path))
            if has_images:
                return run_photo_pipeline(scan_path, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    capture_id = os.path.basename(os.path.abspath(scan_path))

    points = load_capture_point_cloud(scan_path)
    points, corrected_drift = assess_and_correct_drift(points)
    z_floor, z_ceiling, height, ci = estimate_ceiling_height(points)

    rooms, adjacency_graph, total_area = build_multi_room_topology(
        points_or_trajectory=points,
        z_floor=z_floor,
        z_ceiling=z_ceiling,
        tier_name="Tier 3 - LiDAR Walkthrough",
        capture_id=capture_id,
        ci_factor=0.015
    )

    plan = WholePropertyPlan(
        tier="Tier 3 - LiDAR Walkthrough",
        capture_id=capture_id,
        total_area_sq_m=total_area,
        drift_correction_applied=corrected_drift,
        rooms=rooms,
        adjacency_graph=adjacency_graph
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plan.model_dump_json(indent=2))

    print(f"[Tier 3 LiDAR] Reconstructed whole-property plan ({len(rooms)} rooms) written to {output_path}")
    return plan

def main():
    parser = argparse.ArgumentParser(description="Spatial Inspector Property Reconstruction CLI")
    parser.add_argument("--scan", required=True, help="Path to input scan directory or video file")
    parser.add_argument("--output", default="output/property_plan.json", help="Destination JSON path")
    args = parser.parse_args()

    run_reconstruction(args.scan, args.output)

if __name__ == "__main__":
    main()
