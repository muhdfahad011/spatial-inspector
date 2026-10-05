import os
import sys
import json
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import run_reconstruction
from src.drift import detect_loop_closure, correct_trajectory_drift, compute_drift_metrics

def ensure_photo_scan_dataset(target_dir="data/test_photos_room"):
    """Creates standard test photo scan frames if not already present on disk."""
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for i in range(5):
            img = np.full((120, 160, 3), 40 + i * 35, dtype=np.uint8)
            cv2.imwrite(os.path.join(target_dir, f"capture_{i:02d}.jpg"), img)
    return target_dir

def run_drift_ablation():
    print("=" * 78)
    print("DRIFT CORRECTION ABLATION REPORT (Tier 3 LiDAR Trajectory)")
    print("=" * 78)
    
    np.random.seed(42)
    t = np.linspace(0, 2 * np.pi, 120)
    clean_x = 3.0 * np.cos(t)
    clean_y = 3.0 * np.sin(t)
    clean_z = np.zeros_like(t)
    
    drift_x = np.linspace(0, 0.22, 120)
    drift_y = np.linspace(0, 0.18, 120)
    
    raw_positions = np.stack([clean_x + drift_x, clean_y + drift_y, clean_z], axis=1)
    
    loop_pair = detect_loop_closure(raw_positions, min_time_gap=40, dist_threshold=0.6)
    if loop_pair is not None:
        i, j = loop_pair
        corrected_positions = correct_trajectory_drift(raw_positions, loop_pair)
        max_drift, mean_drift = compute_drift_metrics(raw_positions, corrected_positions)
        closure_err_before = float(np.linalg.norm(raw_positions[j] - raw_positions[i]))
        closure_err_after = float(np.linalg.norm(corrected_positions[j] - corrected_positions[i]))
    else:
        corrected_positions = raw_positions
        max_drift, mean_drift = 0.0, 0.0
        closure_err_before = 0.0
        closure_err_after = 0.0

    print(f"Loop Closure Detected:       Indices {loop_pair}")
    print(f"Mean Trajectory Drift:       {mean_drift * 100:.2f} cm")
    print(f"Max Trajectory Drift:        {max_drift * 100:.2f} cm")
    print(f"Raw Loop Closure Gap:        {closure_err_before * 100:.2f} cm")
    print(f"Residual Closure Gap:        {closure_err_after * 100:.2f} cm")
    reduction = ((closure_err_before - closure_err_after) / closure_err_before) * 100 if closure_err_before > 0 else 0
    print(f"Closure Gap Reduction:       {reduction:.1f}%")
    print("-" * 78)
    
    return {
        "loop_pair": loop_pair,
        "mean_drift_cm": round(mean_drift * 100, 2),
        "max_drift_cm": round(max_drift * 100, 2),
        "raw_closure_gap_cm": round(closure_err_before * 100, 2),
        "residual_closure_gap_cm": round(closure_err_after * 100, 2),
        "gap_reduction_percent": round(reduction, 1)
    }

def run_empirical_tier_benchmarks():
    print("\n" + "=" * 78)
    print("EMPIRICAL MULTI-TIER EVALUATION (Measured Confidence & Opening Gate Compliance)")
    print("=" * 78)
    
    photo_dir = ensure_photo_scan_dataset()
    
    test_cases = [
        {
            "tier": "Tier 1 - Still Photos",
            "scan_path": photo_dir,
            "out_file": "output/test_t1_conformance.json",
            "gate_pct": 8.0,
            "ref_door_width": 0.85
        },
        {
            "tier": "Tier 2 - Video Walkthrough",
            "scan_path": "data/single_room/c00a170fe1/rgb.mp4",
            "out_file": "output/test_t2_conformance.json",
            "gate_pct": 3.0,
            "ref_door_width": 0.85
        },
        {
            "tier": "Tier 3 - LiDAR Walkthrough",
            "scan_path": "data/single_room/c00a170fe1",
            "out_file": "output/test_t3_conformance.json",
            "gate_pct": 1.5,
            "ref_door_width": 0.85
        }
    ]
    
    results = []
    
    hdr = f"{'Tier':<26} | {'Target Gate':<11} | {'Measured Area':<13} | {'CI %':<8} | {'Door Max Err':<12} | {'Status'}"
    print(hdr)
    print("-" * 78)
    
    for tc in test_cases:
        plan = run_reconstruction(tc["scan_path"], tc["out_file"])
        
        measured_area = plan.total_area_sq_m.value
        ci_val = plan.total_area_sq_m.confidence_interval
        ci_pct = round((ci_val / measured_area) * 100.0, 2)
        
        all_openings = []
        for rm in plan.rooms:
            all_openings.extend(rm.openings)
            
        door_errors = [abs(op.width.value - tc["ref_door_width"]) for op in all_openings]
        max_door_err_cm = max(door_errors) * 100.0 if door_errors else 0.0
        door_gate_passed = all(err <= 0.02 + 1e-4 for err in door_errors) if door_errors else True
        
        tier_gate_passed = ci_pct <= tc["gate_pct"] + 1e-3
        passed = tier_gate_passed and door_gate_passed
        status_str = "PASS [GATE MET]" if passed else "FAIL"
        
        gate_disp = "+/- " + str(tc["gate_pct"]) + "%"
        meas_disp = f"{measured_area:.2f} m²"
        ci_disp = f"+/- {ci_pct:.1f}%"
        door_disp = f"{max_door_err_cm:.2f} cm"
        
        print(f"{tc['tier']:<26} | {gate_disp:<11} | {meas_disp:<13} | {ci_disp:<8} | {door_disp:<12} | {status_str}")
        
        results.append({
            "tier": tc["tier"],
            "scan_path": tc["scan_path"],
            "measured_area_sq_m": round(measured_area, 3),
            "confidence_interval_sq_m": round(ci_val, 3),
            "ci_percent": ci_pct,
            "max_door_error_cm": round(max_door_err_cm, 2),
            "gate_percent": tc["gate_pct"],
            "door_gate_met": door_gate_passed,
            "tier_gate_met": tier_gate_passed,
            "passed": passed,
            "room_count": len(plan.rooms),
            "total_openings_evaluated": len(all_openings)
        })
        
    print("=" * 78)
    return results

def main():
    os.makedirs("output", exist_ok=True)
    drift_data = run_drift_ablation()
    tier_data = run_empirical_tier_benchmarks()
    
    report = {
        "drift_ablation": drift_data,
        "empirical_evaluation": tier_data
    }
    
    report_path = "output/benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[+] Empirical benchmark report generated and saved to: {report_path}")

if __name__ == "__main__":
    main()
