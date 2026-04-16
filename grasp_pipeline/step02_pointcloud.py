"""
step02_pointcloud.py
=====================
STEP 2 -- POINT CLOUD GENERATION

Responsibilities:
  1. Backproject each pixel from the depth image to 3-D using the pinhole
     camera model and provided intrinsic parameters.
  2. Attach the RGB colour to every 3-D point.
  3. Return a flat (N, 3) point cloud and matching (N, 3) colour array.

Formula
-------
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  Z = depth_raw[v, u]          (depth in metres)

Returns
-------
points : np.ndarray (N, 3)  float32  -- XYZ in metres
colors : np.ndarray (N, 3)  float32  -- RGB in [0, 1]
"""

from __future__ import annotations
import numpy as np
from grasp_pipeline.utils.camera_intrinsics import CameraIntrinsics
from typing import Tuple


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate_pointcloud(
    depth_raw:   np.ndarray,
    rgb_norm:    np.ndarray,
    intrinsics:  CameraIntrinsics,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth image to a coloured 3-D point cloud.

    Parameters
    ----------
    depth_raw  : float32 (H, W) -- depth in metres (original units, after NaN repair)
    rgb_norm   : float32 (H, W, 3) -- RGB in [0, 1]
    intrinsics : CameraIntrinsics -- pinhole camera model

    Returns
    -------
    points : (N, 3) float32 -- valid 3-D points in metres
    colors : (N, 3) float32 -- corresponding RGB colours
    """
    H, W = depth_raw.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy
    scale  = intrinsics.scale           # depth unit -> metres

    # --- Build pixel coordinate grids (vectorised) ------------------------- #
    u_grid = np.arange(W, dtype=np.float32)          # (W,)
    v_grid = np.arange(H, dtype=np.float32)          # (H,)
    uu, vv = np.meshgrid(u_grid, v_grid)              # (H, W) each

    # --- Backproject ------------------------------------------------------- #
    Z = depth_raw / scale                             # (H, W) metres
    X = (uu - cx) * Z / fx                           # (H, W)
    Y = (vv - cy) * Z / fy                           # (H, W)

    # Stack into (H, W, 3)
    xyz = np.stack([X, Y, Z], axis=-1)               # (H, W, 3)

    # --- Filter out invalid depth pixels ----------------------------------- #
    valid_mask = (Z > 0) & np.isfinite(Z)            # (H, W) bool
    points = xyz[valid_mask]                          # (N, 3)
    colors = rgb_norm[valid_mask]                     # (N, 3)

    n_total = H * W
    n_valid = points.shape[0]
    print(
        f"[Step 2] Point cloud generated -- {n_valid:,} / {n_total:,} "
        f"points ({100*n_valid/n_total:.1f}% valid)"
    )
    return points.astype(np.float32), colors.astype(np.float32)
