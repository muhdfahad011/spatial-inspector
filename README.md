# Spatial Inspector: 3D Reconstruction & Insurance Scoping Pipeline

> **Carrier Audit Status**: Verified Compliant (Pydantic v2 Schema Contract Validated)

A professional-grade spatial reconstruction engine that processes raw mobile LiDAR depth streams, camera intrinsics, and pose odometry to generate metric floor plans, opening geometries, and carrier-compliant insurance damage scopes.

---

## Key Capabilities

- **Hardware dToF Unprojection**: Converts raw 16-bit depth frames and camera intrinsics into metric 3D point clouds in global world coordinates.
- **Plane Fitting & Geometry Extraction**: Computes ceiling heights with 95% confidence intervals, slices floor elevations, and extracts wall boundaries and door/window openings.
- **Loop-Closure Drift Mitigation**: Corrects trajectory odometry drift via closed-loop interpolation and planar re-anchoring.
- **Automated Scoping & Concealed Damage Rules**: Maps surface damages directly to standardized insurance repair codes (WTR-DRYW-TSR, WTR-DRYW-DRY) using deterministic building envelope heuristics.
- **Reproducible Part 4 Fix Loop**: Automated test harness comparing uncorrected drift vs. corrected loop closure with quantitative error reporting.

---

## Project Structure

`
spatial-inspector/
|-- data/single_room/        # ARKit LiDAR captures (depth, poses, intrinsics)
|-- docs/
|   |-- COMPLIANCE_MATRIX.md # Part 1: Sensor protocols & carrier standards
|   -- REPORT.md            # Part 3 & 5: Architecture, trade-offs & ablation
|-- output/                  # Generated plan JSONs and fix loop reports
|-- src/
|   |-- schema.py            # Pydantic v2 typed output contract
|   |-- loader.py            # Sensor unprojection & pose transformation
|   |-- geometry.py          # Planar fitting & wall boundary extraction
|   |-- drift.py             # Loop-closure drift correction
|   |-- damage.py            # Damage segmentation & scoping rules
|   |-- pipeline.py          # Unified CLI entry point
|   -- fix_loop.py          # Part 4 reproducible verification loop
|-- requirements.txt         # Pinned Python dependencies
-- README.md
`

---

## Quickstart & Execution

### 1. Environment Setup
`ash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
`

### 2. Run Reconstruction Pipeline
`ash
python -m src.pipeline --scan data/single_room/c00a170fe1 --output output/plan.json
`

### 3. Run Part 4 Reproducible Fix Loop (Before vs. After Ablation)
`ash
python -m src.fix_loop
`
Inspect generated audit metrics in output/fix_loop_report.json.

## Multi-Tier & Conformance Testing

Run the multi-tier test suite:
``powershell
python -m pytest tests/test_multi_tier.py -v
``

### Multi-Tier CLI Execution

``powershell
# Tier 1 (Still Photos)
python -m src.pipeline --scan data/test_photos_room --output output/plan_photos.json

# Tier 2 (Video Walkthrough)
python -m src.pipeline --scan data/single_room/c00a170fe1/rgb.mp4 --output output/plan_video.json

# Tier 3 (LiDAR)
python -m src.pipeline --scan data/single_room/c00a170fe1 --output output/plan_lidar.json
``
