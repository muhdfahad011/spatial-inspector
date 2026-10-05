import os
import numpy as np
import pytest
from src.schema import WholePropertyPlan
from src.pipeline import run_reconstruction
from src.topology import build_multi_room_topology

def test_opening_accuracy_gate():
    """
    Asserts Issue 2 criteria:
    At least 85% of detected openings across all reconstructed rooms 
    must achieve absolute measurement error / confidence bounds <= 2 cm (0.02 m).
    """
    dummy_points = np.array([
        [-2.0, -1.5, 0.0],
        [3.0, 2.5, 2.5],
        [0.0, 0.0, 1.2],
        [1.5, -0.5, 0.8]
    ])
    
    rooms, adjacency, total_area = build_multi_room_topology(
        points_or_trajectory=dummy_points,
        z_floor=0.0,
        z_ceiling=2.45,
        tier_name="Tier 3 - LiDAR Walkthrough",
        capture_id="gate_test_capture",
        ci_factor=0.015
    )
    
    all_openings = []
    for r in rooms:
        all_openings.extend(r.openings)
        
    assert len(all_openings) >= 3, "Expected at least 3 detected openings across multi-room plan"
    
    # Standard doorway ground-truth benchmark
    ground_truth_width = 0.85
    passing_openings = 0
    
    for op in all_openings:
        error = abs(op.width.value - ground_truth_width)
        # Verify both measurement error and sensor confidence interval stay within <= 2 cm (0.02 m)
        if error <= 0.02 and op.width.confidence_interval <= 0.02:
            passing_openings += 1
            
    pass_rate = passing_openings / len(all_openings)
    print(f"\n[Issue 2 Opening Gate] Passing Openings: {passing_openings}/{len(all_openings)} ({pass_rate * 100:.1f}%)")
    assert pass_rate >= 0.85, f"Expected >= 85% openings within <= 2 cm error, got {pass_rate * 100:.1f}%"
