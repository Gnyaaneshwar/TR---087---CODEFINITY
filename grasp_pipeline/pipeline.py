"""
pipeline.py
============
MAIN ORCHESTRATOR -- Autonomous Robotic Grasp Planning Pipeline

Runs all 11 steps in order, enforcing the < 3-second total budget.

Usage (CLI):
    python pipeline.py --rgb data/sample_rgb.png --depth data/sample_depth.png
    python pipeline.py --rgb data/sample_rgb.png --depth data/sample_depth.png --show

Usage (Python API):
    from grasp_pipeline.pipeline import run_pipeline
    result = run_pipeline("data/sample_rgb.png", "data/sample_depth.png")
"""

from __future__ import annotations
import argparse
import time
import warnings
import os
import sys
import numpy as np
from typing import Optional, Union, Dict, Any, List

# -- Step modules --------------------------------------------------------------
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
from grasp_pipeline.utils.camera_intrinsics  import CameraIntrinsics
from grasp_pipeline.utils.grasp_types        import Grasp

# -----------------------------------------------------------------------------
# Pipeline constants (configurable)
# -----------------------------------------------------------------------------
N_CANDIDATES = 50       # total grasp candidates (>= 20)
TOP_K        = 10       # final top grasps returned
TIME_BUDGET  = 3.0      # seconds -- warn if exceeded


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_pipeline(
    rgb_input:    Union[str, np.ndarray],
    depth_input:  Union[str, np.ndarray],
    intrinsics:   Optional[CameraIntrinsics] = None,
    output_dir:   str  = "outputs",
    n_candidates: int  = N_CANDIDATES,
    top_k:        int  = TOP_K,
    show_viewer:  bool = False,
) -> Dict[str, Any]:
    """
    Run the full 11-step grasp planning pipeline.

    Parameters
    ----------
    rgb_input    : file path OR (H,W,3) uint8/float32 array
    depth_input  : file path OR (H,W)   float32 array
    intrinsics   : CameraIntrinsics -- defaults to 640x480 Kinect-style
    output_dir   : where to save JSON + PNG
    n_candidates : total grasp candidates to generate (>= 20)
    top_k        : final top-K to select and return
    show_viewer  : open interactive Open3D window after rendering

    Returns
    -------
    dict with keys:
        "top_grasps"   : List[Grasp]
        "output_path"  : str (JSON file)
        "vis_path"     : str (PNG file)
        "result_dict"  : dict (JSON payload)
        "timings"      : dict[step_name -> seconds]
    """
    pipeline_start = time.perf_counter()
    timings: Dict[str, float] = {}

    if intrinsics is None:
        intrinsics = CameraIntrinsics()

    _banner("AUTONOMOUS ROBOTIC GRASP PLANNING PIPELINE")
    # force UTF-8 output on Windows
    import sys, io
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # -- STEP 1: Input Handling ---------------------------------------------
    t = time.perf_counter()
    rgb_norm, depth_norm, depth_raw = load_and_validate(rgb_input, depth_input)
    timings["step01_input"] = time.perf_counter() - t

    # Adapt intrinsics to actual image size
    H, W = rgb_norm.shape[:2]
    intrinsics.width  = W
    intrinsics.height = H
    if intrinsics.cx == 319.5 and W != 640:
        intrinsics.cx = W  / 2.0 - 0.5
        intrinsics.cy = H  / 2.0 - 0.5

    # -- STEP 2: Point Cloud Generation ----------------------------------------
    # depth_raw is in mm (from PNG uint16 storage); convert to metres
    depth_metres = depth_raw / 1000.0
    t = time.perf_counter()
    points, colors = generate_pointcloud(depth_metres, rgb_norm, intrinsics)
    timings["step02_pointcloud"] = time.perf_counter() - t

    # -- STEP 3: Scene Segmentation ----------------------------------------
    t = time.perf_counter()
    P_filtered, C_filtered, _ = segment_scene(points, colors)
    timings["step03_segmentation"] = time.perf_counter() - t

    if P_filtered.shape[0] < 10:
        warnings.warn("[Pipeline] Very few foreground points -- using full cloud.")
        P_filtered, C_filtered = points, colors

    # -- STEP 4: Grasp Candidate Generation --------------------------------
    t = time.perf_counter()
    grasps: List[Grasp] = generate_grasp_candidates(
        P_filtered, n_candidates=n_candidates
    )
    timings["step04_candidates"] = time.perf_counter() - t

    total_candidates = len(grasps)

    # -- STEP 5: Initial Scoring -------------------------------------------
    t = time.perf_counter()
    grasps = score_initial(grasps, P_filtered)
    timings["step05_scoring"] = time.perf_counter() - t

    # -- STEP 6: Collision Checking ----------------------------------------
    t = time.perf_counter()
    grasps = check_collisions(grasps, P_filtered)
    timings["step06_collision"] = time.perf_counter() - t

    # -- STEP 7: Stability Estimation --------------------------------------
    t = time.perf_counter()
    grasps = estimate_stability(grasps, P_filtered)
    timings["step07_stability"] = time.perf_counter() - t

    # -- STEP 8: Hybrid Re-Ranking -----------------------------------------
    t = time.perf_counter()
    top_grasps = rank_and_select(grasps, top_k=top_k)
    timings["step08_ranking"] = time.perf_counter() - t

    # -- STEP 9: Physics Validation ----------------------------------------
    t = time.perf_counter()
    top_grasps = validate_physics(top_grasps)
    timings["step09_physics"] = time.perf_counter() - t

    # -- STEP 10: Output Generation ----------------------------------------
    t = time.perf_counter()
    output_path, result_dict = generate_output(
        top_grasps, total_candidates,
        output_dir=output_dir,
        extra_metadata={"timings_s": {k: round(v, 4) for k, v in timings.items()}},
    )
    timings["step10_output"] = time.perf_counter() - t

    # -- STEP 11: Visualization --------------------------------------------
    t = time.perf_counter()
    vis_path = visualize(
        P_filtered, C_filtered, top_grasps,
        output_dir=output_dir, show=show_viewer
    )
    timings["step11_visualize"] = time.perf_counter() - t

    # -- Summary -----------------------------------------------------------
    total_time = time.perf_counter() - pipeline_start
    timings["total"] = total_time

    _print_timing_table(timings)

    if total_time > TIME_BUDGET:
        warnings.warn(
            f"[Pipeline] Total time {total_time:.2f}s exceeds "
            f"{TIME_BUDGET}s budget.", stacklevel=2
        )
    else:
        print(f"\nOK Pipeline completed in {total_time:.3f}s (budget: {TIME_BUDGET}s)")

    return {
        "top_grasps":   top_grasps,
        "output_path":  output_path,
        "vis_path":     vis_path,
        "result_dict":  result_dict,
        "timings":      timings,
    }


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Robotic Grasp Planning Pipeline (RGB-D -> Top Grasps)"
    )
    parser.add_argument("--rgb",   required=True,  help="Path to RGB image")
    parser.add_argument("--depth", required=True,  help="Path to depth image")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES,
                        help="Number of grasp candidates (>= 20)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Final top grasps to select")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive 3-D viewer after rendering")
    parser.add_argument("--fx", type=float, default=525.0)
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)

    args = parser.parse_args()

    intrinsics = CameraIntrinsics(
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
    )

    run_pipeline(
        rgb_input    = args.rgb,
        depth_input  = args.depth,
        intrinsics   = intrinsics,
        output_dir   = args.output_dir,
        n_candidates = args.n_candidates,
        top_k        = args.top_k,
        show_viewer  = args.show,
    )


# -----------------------------------------------------------------------------
# Display helpers
# -----------------------------------------------------------------------------

def _banner(title: str) -> None:
    w = 62
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def _print_timing_table(timings: Dict[str, float]) -> None:
    print("\n+---------------------------------+----------+")
    print("| Step                            |   Time   |")
    print("+---------------------------------+----------+")
    for step, t in timings.items():
        if step == "total":
            continue
        label = step.replace("_", " ").title()
        bar   = "#" * int(t * 40 / max(timings["total"], 0.01))
        print(f"| {label:<31} | {t:>6.3f}s |")
    print("+---------------------------------+----------+")
    print(f"| {'TOTAL':<31} | {timings['total']:>6.3f}s |")
    print("+---------------------------------+----------+")


if __name__ == "__main__":
    _cli()
