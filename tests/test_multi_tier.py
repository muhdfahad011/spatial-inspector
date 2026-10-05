import os
import pytest
import numpy as np
import cv2
from src.schema import WholePropertyPlan
from src.photo_pipeline import run_photo_pipeline
from src.video_pipeline import run_video_pipeline
from src.pipeline import run_reconstruction

def test_tier1_photos_conformance(tmp_path):
    target_dir = "data/test_photos_room"
    if not os.path.isdir(target_dir):
        # Create minimal synthetic photo set if local dataset is absent
        target_dir = str(tmp_path / "synthetic_photos")
        os.makedirs(target_dir, exist_ok=True)
        for i in range(4):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Wall {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(target_dir, f"frame_{i:02d}.jpg"), img)

    out_file = str(tmp_path / "test_t1_conformance.json")
    plan = run_photo_pipeline(target_dir, out_file)
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

def test_tier2_video_conformance(tmp_path):
    target_video = "data/single_room/c00a170fe1/rgb.mp4"
    if not os.path.isfile(target_video):
        # Create minimal valid synthetic video walkthrough if local dataset is absent
        target_video = str(tmp_path / "synthetic_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(target_video, fourcc, 10.0, (320, 240))
        for f in range(20):
            frame = np.full((240, 320, 3), (f * 12) % 255, dtype=np.uint8)
            cv2.circle(frame, (40 + f * 10, 120), 20, (255, 255, 255), -1)
            out.write(frame)
        out.release()

    out_file = str(tmp_path / "test_t2_conformance.json")
    plan = run_video_pipeline(target_video, out_file)
    assert isinstance(plan, WholePropertyPlan)
    assert plan.tier == "Tier 2 - Video Walkthrough"

    room = plan.rooms[0]
    expected_ci = round(room.floor_area_sq_m.value * 0.03, 3)
    assert abs(room.floor_area_sq_m.confidence_interval - expected_ci) < 1e-2

    assert len(room.damages) > 0
    assert room.damages[0].scope_line_item == "WTR-DRYW-DRY"

@pytest.mark.skipif(not os.path.exists("data/single_room/c00a170fe1"), reason="LiDAR sensor dataset requires local capture files")
def test_tier3_lidar_conformance(tmp_path):
    out_file = str(tmp_path / "test_t3_conformance.json")
    plan = run_reconstruction("data/single_room/c00a170fe1", out_file)
    assert isinstance(plan, WholePropertyPlan)
    assert "Tier 3" in plan.tier
    assert plan.drift_correction_applied is True

    room = plan.rooms[0]
    for wall in room.walls:
        assert wall.length.confidence_interval <= 0.02

    assert room.ceiling_height.confidence_interval <= 0.10
