# Spatial Capture Protocol & Insurance Carrier Compliance Matrix

## 1. Multi-Tier Capture Protocol Comparison

| Metric / Requirement | Tier 1: Hardware-Synchronized LiDAR Stream | Tier 2: Monocular Video Walkthrough (SfM/SLAM) | Tier 3: Sparse Multi-View Photos |
| :--- | :--- | :--- | :--- |
| **Primary Sensors** | Direct Time-of-Flight (dToF) LiDAR + Wide RGB + 60Hz IMU | High-Res Monocular RGB (4K/60fps) + 100Hz VIO | Uncalibrated Mobile Phone / DSLR Still Photos |
| **Metric Scale Recovery** | Direct physical measurement (hardware unprojected depth) | Scale-ambiguous without known fiducials / ground-truth metric scale bars | Requires collinear homography or user-annotated reference target |
| **Positional Accuracy** | Sub-centimeter (< 10 mm error across a 5m span) | 2–5 cm relative error; prone to drift over long corridors | 5–15 cm; heavy reprojection error on oblique surfaces |
| **Loop-Closure Sensitivity** | Low-to-moderate drift; solved via pose graph & plane re-anchoring | High drift; severe trajectory skew without loop closure | Extremely high; point cloud stitching fails without wide overlap (>70%) |
| **Carrier Audit Acceptance** | **Automated Acceptance** (95% CI bounds reported per room dimension) | **Conditional Acceptance** (Requires secondary metric verification) | **Manual Adjuster Review** (High rejection rate for structural scope) |

---

## 2. Sensor Failure Modes & Mitigation Matrix

### A. Specular & Transparent Surfaces (Mirrors, Glass Showers, Windows)
* **Failure Mechanism**: LiDAR laser pulses either penetrate glass entirely (ranging objects outside the structural envelope) or bounce off reflective surfaces, generating "phantom" point clusters behind walls.
* **Algorithmic Mitigation**: 
  1. Surface normal clustering: Points behind coplanar wall fits are flagged as specular noise.
  2. Multi-modal RGB cross-validation: Edge detection on RGB frames confirms physical window/mirror boundaries, bounding depth penetration.

### B. Textureless Surfaces (White Drywall, Flat Ceilings)
* **Failure Mechanism**: Classical feature extractors (SIFT, ORB) fail to track visual landmarks, leading to visual odometry track loss.
* **Algorithmic Mitigation**:
  1. Hardware dToF unprojection bypasses photometric tracking dependencies.
  2. Continuous IMU dead-reckoning bridges short camera dropouts until planar structural boundaries are recovered.

### C. Low-Light & Shadow Gradients (Unlit Basements, Attics)
* **Failure Mechanism**: Severe photometric sensor noise degrades monocular feature matches and edge definition.
* **Algorithmic Mitigation**:
  1. Active infrared illumination from the LiDAR sensor maintains depth accuracy regardless of ambient lumens.
  2. Histogram-based elevation slicing preserves ceiling/floor detection even with zero valid RGB texture.

---

## 3. Insurance Carrier Scoping Acceptance Rules

1. **Explicit Uncertainty Reporting**: Every measurement (length, area, ceiling height) must include a 95% Confidence Interval ($\pm \delta$). Single-value scalars without error margins are rejected by modern audit engines.
2. **Concealed Damage Heuristics**: Moisture migration follows physical gravity and capillary action. Any surface damage exceeding $1.5\text{ m}^2$ or reaching within 10 cm of the floor line must trigger stud-bay and subfloor scoping line items (`WTR-DRYW-TSR`).
3. **Reproducible Traceability**: Every line-item estimate must link back to a persistent surface identifier and coordinate bounding box within the spatial reconstruction graph.
