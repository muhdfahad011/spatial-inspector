import pytest
import numpy as np
from src.schema import WholePropertyPlan
from src.photo_pipeline import run_photo_pipeline
from src.video_pipeline import run_video_pipeline
from src.pipeline import run_reconstruction

def test_tier1_photos_conformance():
    plan = run_photo_pipeline("data/test_photos_room", "output/test_t1_conformance.json")
    assert isinstance(plan, WholePropertyPlan)
    assert plan.tier == "Tier 1 - Still Photos"
    assert len(plan.rooms) == 1
    
    room = plan.rooms[0]
    expected_ci = round(room.floor_area_sq_m.value * 0.08, 3)
    assert abs(room.floor_area_sq_m.confidence_interval - expected_ci) < 1e-2
    
    walls = room.walls
    assert len(walls) == 4
    for i in range(len(walls)):
        curr_end = walls[i].end_point[:2]
        next_start = walls[(i + 1) % len(walls)].start_point[:2]
        assert np.allclose(curr_end, next_start, atol=1e-3)

def test_tier2_video_conformance():
    plan = run_video_pipeline("data/single_room/c00a170fe1/rgb.mp4", "output/test_t2_conformance.json")
    assert isinstance(plan, WholePropertyPlan)
    assert plan.tier == "Tier 2 - Video Walkthrough"
    assert plan.capture_id == "c00a170fe1"
    
    room = plan.rooms[0]
    expected_ci = round(room.floor_area_sq_m.value * 0.03, 3)
    assert abs(room.floor_area_sq_m.confidence_interval - expected_ci) < 1e-2
    
    assert len(room.damages) > 0
    assert room.damages[0].scope_line_item == "WTR-DRYW-DRY"

def test_tier3_lidar_conformance():
    plan = run_reconstruction("data/single_room/c00a170fe1", "output/test_t3_conformance.json")
    assert isinstance(plan, WholePropertyPlan)
    assert "Tier 3" in plan.tier
    assert plan.drift_correction_applied is True
    
    room = plan.rooms[0]
    # Verify Tier 3 LiDAR millimeter/centimeter level CI precision on structural walls
    for wall in room.walls:
        assert wall.length.confidence_interval <= 0.02  # <= 2cm specification
    
    # Statistical ceiling height estimation uncertainty bound (sub-decimeter)
    assert room.ceiling_height.confidence_interval <= 0.10
