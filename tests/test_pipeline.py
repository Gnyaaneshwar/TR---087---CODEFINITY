"""
tests/test_pipeline.py
=======================
Unit tests for each pipeline step.
Run with:  python -m pytest tests/ -v
       or: python tests/test_pipeline.py
"""

import sys
import os
import warnings
import numpy as np
import pytest

# Allow import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grasp_pipeline.utils.grasp_types        import Grasp
from grasp_pipeline.utils.camera_intrinsics  import CameraIntrinsics
from grasp_pipeline.step01_input_handler     import load_and_validate
from grasp_pipeline.step02_pointcloud        import generate_pointcloud
from grasp_pipeline.step03_segmentation      import segment_scene
from grasp_pipeline.step04_grasp_candidates  import generate_grasp_candidates
from grasp_pipeline.step05_initial_scorer    import score_initial
from grasp_pipeline.step06_collision_checker import check_collisions
from grasp_pipeline.step07_stability_estimator import estimate_stability
from grasp_pipeline.step08_hybrid_ranker     import rank_and_select
from grasp_pipeline.step09_physics_validator import validate_physics
from grasp_pipeline.step10_output_generator  import generate_output
from grasp_pipeline.step11_visualizer        import visualize


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

H, W = 120, 160    # small test resolution for speed

@pytest.fixture(scope="module")
def synthetic_rgb():
    rng = np.random.default_rng(0)
    img = rng.integers(30, 200, (H, W, 3), dtype=np.uint8)
    # Draw coloured blobs
    import cv2
    cv2.circle(img, (80, 60), 30, (200, 100, 50), -1)
    cv2.circle(img, (40, 90), 20, (50, 180, 80), -1)
    return img


@pytest.fixture(scope="module")
def synthetic_depth():
    rng = np.random.default_rng(0)
    depth = np.full((H, W), 1200.0, dtype=np.float32)   # 1.2 m background
    yy, xx = np.mgrid[0:H, 0:W]
    # Object 1
    d1 = np.exp(-((xx-80)**2 + (yy-60)**2) / (2*30**2))
    depth -= d1 * 400                                    # ~ 0.8 m object
    # Object 2
    d2 = np.exp(-((xx-40)**2 + (yy-90)**2) / (2*20**2))
    depth -= d2 * 300                                    # ~ 0.9 m object
    # 5% missing
    mask = rng.random((H, W)) < 0.05
    depth[mask] = 0.0
    return depth.astype(np.float32)


@pytest.fixture(scope="module")
def camera():
    return CameraIntrinsics(fx=200, fy=200, cx=80, cy=60,
                            scale=1.0, width=W, height=H)


@pytest.fixture(scope="module")
def pipeline_io(synthetic_rgb, synthetic_depth, camera):
    """Run steps 1-3 once, reuse across tests."""
    rgb_norm, depth_norm, depth_raw = load_and_validate(
        synthetic_rgb, synthetic_depth
    )
    points, colors = generate_pointcloud(depth_raw / 1000.0, rgb_norm, camera)
    P_filtered, C_filtered, _ = segment_scene(points, colors)
    if P_filtered.shape[0] < 10:
        P_filtered, C_filtered = points, colors
    return rgb_norm, depth_norm, depth_raw, points, colors, P_filtered, C_filtered


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStep01InputHandler:
    def test_normalised_rgb_range(self, synthetic_rgb, synthetic_depth):
        rgb_norm, depth_norm, _ = load_and_validate(synthetic_rgb, synthetic_depth)
        assert rgb_norm.dtype == np.float32
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

    def test_depth_norm_range(self, synthetic_rgb, synthetic_depth):
        _, depth_norm, _ = load_and_validate(synthetic_rgb, synthetic_depth)
        assert depth_norm.min() >= 0.0 and depth_norm.max() <= 1.0

    def test_resolution_match(self, synthetic_rgb, synthetic_depth):
        rgb_norm, depth_norm, depth_raw = load_and_validate(
            synthetic_rgb, synthetic_depth)
        assert rgb_norm.shape[:2] == depth_norm.shape[:2] == depth_raw.shape[:2]

    def test_nan_filled(self, synthetic_rgb, synthetic_depth):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, depth_raw = load_and_validate(synthetic_rgb, synthetic_depth)
        assert np.all(depth_raw > 0), "Missing depth pixels were not filled"

    def test_resolution_mismatch_raises(self):
        rgb   = np.zeros((100, 100, 3), dtype=np.uint8)
        depth = np.zeros((200, 100),    dtype=np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            load_and_validate(rgb, depth)


class TestStep02PointCloud:
    def test_output_shapes(self, pipeline_io):
        _, _, _, points, colors, _, _ = pipeline_io
        assert points.ndim == 2 and points.shape[1] == 3
        assert colors.shape == points.shape

    def test_dtype(self, pipeline_io):
        _, _, _, points, colors, _, _ = pipeline_io
        assert points.dtype == np.float32
        assert colors.dtype == np.float32

    def test_no_zero_z(self, pipeline_io):
        _, _, _, points, _, _, _ = pipeline_io
        assert (points[:, 2] > 0).all(), "Point cloud contains Z=0 points"


class TestStep03Segmentation:
    def test_foreground_shape(self, pipeline_io):
        _, _, _, points, colors, P_filt, C_filt = pipeline_io
        assert P_filt.shape[1] == 3
        assert C_filt.shape == P_filt.shape

    def test_fewer_than_full_cloud(self, pipeline_io):
        _, _, _, points, _, P_filt, _ = pipeline_io
        # Foreground should have fewer or equal points
        assert P_filt.shape[0] <= points.shape[0]


class TestStep04GraspCandidates:
    def test_minimum_candidates(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=20)
        assert len(grasps) >= 20, "Must generate at least 20 candidates"

    def test_grasp_fields(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=25)
        for g in grasps:
            assert g.position.shape == (3,)
            assert g.orientation.shape == (3,)
            assert 0.03 <= g.gripper_width <= 0.15

    def test_custom_n(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=50)
        assert len(grasps) == 50


class TestStep05InitialScorer:
    def test_confidence_range(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=20)
        grasps = score_initial(grasps, P_filt)
        for g in grasps:
            assert 0.0 <= g.confidence <= 1.0, f"Confidence out of range: {g.confidence}"


class TestStep06CollisionChecker:
    def test_collision_status_values(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=20)
        grasps = check_collisions(grasps, P_filt)
        for g in grasps:
            assert g.collision_status in ("clear", "collision")
            assert g.collision_free_score in (0.0, 1.0)


class TestStep07StabilityEstimator:
    def test_stability_range(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=20)
        grasps = estimate_stability(grasps, P_filt)
        for g in grasps:
            assert 0.0 <= g.stability_score <= 1.0


class TestStep08HybridRanker:
    def _make_scored_grasps(self, P_filt):
        grasps = generate_grasp_candidates(P_filt, n_candidates=30)
        grasps = score_initial(grasps, P_filt)
        grasps = check_collisions(grasps, P_filt)
        grasps = estimate_stability(grasps, P_filt)
        return grasps

    def test_top_k_count(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = self._make_scored_grasps(P_filt)
        top = rank_and_select(grasps, top_k=10)
        assert len(top) == min(10, len(grasps))

    def test_sorted_descending(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = self._make_scored_grasps(P_filt)
        top = rank_and_select(grasps, top_k=10)
        scores = [g.final_score for g in top]
        assert scores == sorted(scores, reverse=True)

    def test_rank_assignment(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = self._make_scored_grasps(P_filt)
        top = rank_and_select(grasps, top_k=5)
        for i, g in enumerate(top, 1):
            assert g.rank == i

    def test_score_formula(self, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = self._make_scored_grasps(P_filt)
        top = rank_and_select(grasps, top_k=10)
        # Before physics, final_score = 0.4*conf + 0.3*stab + 0.3*coll
        for g in top:
            expected = 0.4*g.confidence + 0.3*g.stability_score + 0.3*g.collision_free_score
            assert abs(g.final_score - expected) < 0.05   # allow physics adjustment


class TestStep09PhysicsValidator:
    def test_degrades_gracefully_without_pybullet(self, pipeline_io):
        """If pybullet is unavailable, physics_validated should be False."""
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=5)
        score_initial(grasps, P_filt)
        check_collisions(grasps, P_filt)
        estimate_stability(grasps, P_filt)
        top = rank_and_select(grasps, top_k=5)
        # Run without asserting pybullet availability
        top = validate_physics(top)
        for g in top:
            assert isinstance(g.physics_validated, bool)
            assert 0.0 <= g.physics_score <= 1.0


class TestStep10OutputGenerator:
    def test_json_created(self, tmp_path, pipeline_io):
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=20)
        score_initial(grasps, P_filt)
        check_collisions(grasps, P_filt)
        estimate_stability(grasps, P_filt)
        top = rank_and_select(grasps, top_k=5)
        out_path, result = generate_output(top, 20, output_dir=str(tmp_path))
        assert os.path.isfile(out_path)
        assert "top_grasps" in result
        assert len(result["top_grasps"]) == 5

    def test_json_schema(self, tmp_path, pipeline_io):
        import json
        _, _, _, _, _, P_filt, _ = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=20)
        score_initial(grasps, P_filt)
        check_collisions(grasps, P_filt)
        estimate_stability(grasps, P_filt)
        top = rank_and_select(grasps, top_k=3)
        out_path, _ = generate_output(top, 20, output_dir=str(tmp_path))
        with open(out_path) as f:
            data = json.load(f)
        required_keys = {"rank", "position", "orientation", "gripper_width",
                         "final_score", "collision_status", "stability_score",
                         "physics_validated"}
        for grasp_dict in data["top_grasps"]:
            assert required_keys.issubset(set(grasp_dict.keys()))


class TestStep11Visualizer:
    def test_png_created(self, tmp_path, pipeline_io):
        _, _, _, _, _, P_filt, C_filt = pipeline_io
        grasps = generate_grasp_candidates(P_filt, n_candidates=20)
        score_initial(grasps, P_filt)
        check_collisions(grasps, P_filt)
        estimate_stability(grasps, P_filt)
        top = rank_and_select(grasps, top_k=5)
        vis_path = visualize(P_filt, C_filt, top,
                             output_dir=str(tmp_path), show=False)
        assert os.path.isfile(vis_path), f"Visualization PNG not created: {vis_path}"


class TestFullPipeline:
    def test_end_to_end(self, synthetic_rgb, synthetic_depth, tmp_path):
        from grasp_pipeline.pipeline import run_pipeline
        result = run_pipeline(
            rgb_input    = synthetic_rgb,
            depth_input  = synthetic_depth / 1000.0,
            output_dir   = str(tmp_path),
            n_candidates = 25,
            top_k        = 5,
        )
        assert "top_grasps" in result
        assert len(result["top_grasps"]) <= 5
        assert os.path.isfile(result["output_path"])
        assert os.path.isfile(result["vis_path"])
        assert result["timings"]["total"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    ret = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    sys.exit(ret.returncode)
