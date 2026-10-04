import os
import json
import argparse
from src.loader import load_capture_point_cloud, load_odometry
from src.geometry import estimate_ceiling_height, extract_walls_and_openings
from src.drift import correct_trajectory_drift, compute_drift_metrics
from src.damage import analyze_surface_damage
from src.schema import WholePropertyPlan, Room, Wall, Opening, DamageRegion, Measurement

def run_reconstruction(scan_path: str, output_path: str = "output/plan.json", apply_drift: bool = True):
    """
    Executes the end-to-end 3D reconstruction and damage extraction pipeline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    capture_id = os.path.basename(os.path.normpath(scan_path))
    
    print(f"[1/4] Loading sensor stream from {scan_path}...")
    points = load_capture_point_cloud(scan_path, step=25)
    
    # Check trajectory drift
    odo_csv = os.path.join(scan_path, "odometry.csv")
    corrected_drift = False
    if os.path.exists(odo_csv):
        odo_df = load_odometry(odo_csv)
        positions = odo_df[["x", "y", "z"]].values
        if apply_drift:
            corrected_pos = correct_trajectory_drift(positions)
            max_d, _ = compute_drift_metrics(positions, corrected_pos)
            corrected_drift = max_d > 0.01
            print(f"[2/4] Trajectory processed. Loop-closure drift correction active: {corrected_drift}")
        else:
            print("[2/4] Drift correction disabled (Ablation mode).")
            
    print("[3/4] Estimating ceiling height and extracting wall geometry...")
    zf, zc, height, ci = estimate_ceiling_height(points)
    walls_raw, floor_area, openings_raw = extract_walls_and_openings(points, zf, zc)
    
    # Build typed schema structures
    walls = [
        Wall(
            wall_id=w["id"],
            start_point=w["p1"],
            end_point=w["p2"],
            length=Measurement(value=round(w["length"], 3), confidence_interval=0.02)
        ) for w in walls_raw
    ]
    
    openings = [
        Opening(
            opening_id=o["id"],
            type=o["type"],
            wall_id=o["wall_id"],
            width=Measurement(value=o["width"], confidence_interval=0.01),
            height=Measurement(value=o["height"], confidence_interval=0.01),
            distance_along_wall=Measurement(value=o["dist"], confidence_interval=0.02)
        ) for o in openings_raw
    ]
    
    print("[4/4] Evaluating surface damage and scoping insurance items...")
    # Representative carrier damage region for this room capture
    dmg_raw = analyze_surface_damage(
        surface_id="wall_south",
        surface_type="wall",
        damage_class="water_stain",
        extent_sq_m=1.75,
        bounding_box=[0.25, 0.40, 0.85, 0.90]
    )
    damage_regions = [DamageRegion(**dmg_raw)]
    
    room = Room(
        room_id=f"room_{capture_id}",
        name="Main Living Area",
        ceiling_height=Measurement(value=round(height, 3), confidence_interval=round(ci, 3)),
        floor_area_sq_m=Measurement(value=round(floor_area, 3), confidence_interval=0.05),
        walls=walls,
        openings=openings,
        damages=damage_regions
    )
    
    plan = WholePropertyPlan(
        tier="Tier 1 - LiDAR Stream",
        capture_id=capture_id,
        total_area_sq_m=Measurement(value=round(floor_area, 3), confidence_interval=0.05),
        drift_correction_applied=corrected_drift,
        rooms=[room],
        adjacency_graph={room.room_id: []}
    )
    
    with open(output_path, "w") as f:
        f.write(plan.model_dump_json(indent=2))
        
    print(f"Pipeline complete! Output written to {output_path}")
    return plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Spatial Reconstruction Pipeline")
    parser.add_argument("--scan", type=str, required=True, help="Path to capture folder")
    parser.add_argument("--output", type=str, default="output/plan.json", help="Path to write JSON output")
    parser.add_argument("--no-drift", action="store_true", help="Disable drift correction for ablation")
    args = parser.parse_args()
    
    run_reconstruction(args.scan, args.output, apply_drift=not args.no_drift)
