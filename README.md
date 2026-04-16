# Autonomous Robotic Grasp Planning Pipeline

A complete, modular 11-step pipeline for robotic grasp planning from a single RGB-D image of a cluttered bin.

## Features

| Step | Module | Description |
|------|--------|-------------|
| 1 | `step01_input_handler` | Load, validate, NaN-repair, normalise RGB + depth |
| 2 | `step02_pointcloud` | Pinhole backprojection → 3D point cloud |
| 3 | `step03_segmentation` | Depth-threshold + DBSCAN foreground segmentation |
| 4 | `step04_grasp_candidates` | Surface-normal analytical grasp generation (≥20 candidates) |
| 5 | `step05_initial_scorer` | Density + normal-consistency + centroid confidence scoring |
| 6 | `step06_collision_checker` | Oriented bounding-box gripper collision detection |
| 7 | `step07_stability_estimator` | Surface alignment + contact symmetry stability scoring |
| 8 | `step08_hybrid_ranker` | `0.4×conf + 0.3×stability + 0.3×collision` → top-10 |
| 9 | `step09_physics_validator` | PyBullet headless lift/slip simulation |
| 10 | `step10_output_generator` | JSON export with full grasp metadata |
| 11 | `step11_visualizer` | Open3D 3D render (matplotlib fallback) with colour-coded grasps |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate synthetic test data
```bash
python data/generate_test_data.py
```

### 2. Run the full pipeline
```bash
python -m grasp_pipeline.pipeline --rgb data/sample_rgb.png --depth data/sample_depth.png
```

### 3. Run with interactive viewer
```bash
python -m grasp_pipeline.pipeline --rgb data/sample_rgb.png --depth data/sample_depth.png --show
```

### 4. Use as Python API
```python
from grasp_pipeline.pipeline import run_pipeline

result = run_pipeline(
    rgb_input    = "data/sample_rgb.png",
    depth_input  = "data/sample_depth.png",
    n_candidates = 50,
    top_k        = 10,
)

print(f"Top grasp: {result['top_grasps'][0]}")
print(f"JSON:      {result['output_path']}")
print(f"Image:     {result['vis_path']}")
```

## Output

### JSON (`outputs/grasps_<timestamp>.json`)
```json
{
  "metadata": { "total_candidates": 50, ... },
  "top_grasps": [
    {
      "rank": 1,
      "position": {"x": 0.12, "y": -0.05, "z": 0.82},
      "orientation": {"roll": 0.0, "pitch": 1.57, "yaw": 0.3},
      "gripper_width": 0.07,
      "confidence": 0.91,
      "stability_score": 0.85,
      "collision_free_score": 1.0,
      "final_score": 0.895,
      "collision_status": "clear",
      "physics_validated": true,
      "physics_score": 0.92
    }
  ]
}
```

### Visualization (`outputs/visualization_<timestamp>.png`)
- 🟢 **Green** — Best grasp (rank 1)
- 🟡 **Yellow** — Valid grasps
- 🔴 **Red** — Collision grasps
- Each grasp shows XYZ axes (R/G/B) and gripper lines

## Camera Intrinsics (CLI)
```bash
python -m grasp_pipeline.pipeline \
  --rgb data/sample_rgb.png \
  --depth data/sample_depth.png \
  --fx 617 --fy 617 --cx 320 --cy 240
```

## Running Tests
```bash
python -m pytest tests/ -v
```

## Project Structure
```
tensor26/
├── grasp_pipeline/
│   ├── pipeline.py              # Main orchestrator + CLI
│   ├── step01_input_handler.py
│   ├── step02_pointcloud.py
│   ├── step03_segmentation.py
│   ├── step04_grasp_candidates.py
│   ├── step05_initial_scorer.py
│   ├── step06_collision_checker.py
│   ├── step07_stability_estimator.py
│   ├── step08_hybrid_ranker.py
│   ├── step09_physics_validator.py
│   ├── step10_output_generator.py
│   ├── step11_visualizer.py
│   └── utils/
│       ├── grasp_types.py       # Grasp dataclass
│       └── camera_intrinsics.py
├── data/
│   └── generate_test_data.py
├── outputs/                     # JSON + PNG outputs (auto-created)
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Notes
- **GraspNet model**: The analytical generator mirrors GraspNet's output schema. Set `USE_GRASPNET=True` in `pipeline.py` to gate a real model.
- **PyBullet**: If not installed, Step 9 gracefully degrades (`physics_validated=False`).
- **Performance**: Targets < 3s on CPU using vectorised NumPy + KD-tree operations.
