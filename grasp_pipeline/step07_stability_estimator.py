"""
step07_stability_estimator.py
==============================
STEP 7 -- STABILITY ESTIMATION

Responsibilities:
  1. For each remaining grasp, fit a local plane to nearby points.
  2. Compute the alignment angle between gripper approach vector and surface normal.
  3. Compute contact symmetry score (are both contact points equidistant?).
  4. stability_score = cos(alignment_angle) x symmetry_score
     -> 1.0 = perfect perpendicular approach, symmetric contacts
     -> 0.0 = tangential approach or asymmetric

Returns
-------
List[Grasp]  -- grasps with `stability_score` populated in [0, 1]
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from typing import List

from grasp_pipeline.utils.grasp_types import Grasp


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def estimate_stability(
    grasps:    List[Grasp],
    points:    np.ndarray,
    radius:    float = 0.04,    # local plane fitting radius (m)
    k_min:     int   = 6,       # minimum neighbours for plane fit
) -> List[Grasp]:
    """
    Estimate grasp stability from surface geometry.

    Parameters
    ----------
    grasps  : List[Grasp]   -- candidates with position and orientation
    points  : (M, 3) float32 -- foreground point cloud
    radius  : float          -- neighbourhood radius for plane fitting
    k_min   : int            -- minimum k-NN count to attempt plane fit

    Returns
    -------
    List[Grasp]  -- grasps with `stability_score` ∈ [0, 1] set
    """
    tree = cKDTree(points)

    for g in grasps:
        pos = g.position.astype(np.float64)

        # ---- 1. Fit local plane ------------------------------------------ #
        idx_ball = tree.query_ball_point(pos, r=radius)

        if len(idx_ball) < k_min:
            # Not enough neighbours -> fall back to k-NN
            k_fallback = max(k_min, 10)
            _, idx_ball = tree.query(pos, k=k_fallback)

        neighbours = points[idx_ball].astype(np.float64)    # (K, 3)
        surface_normal = _fit_plane_normal(neighbours)       # unit vector

        # ---- 2. Approach vector from grasp orientation ------------------- #
        R        = _rpy_to_rotation_matrix(g.orientation)
        approach = R[:, 2].astype(np.float64)               # gripper Z-axis

        # ---- 3. Alignment score ------------------------------------------ #
        cos_angle      = float(np.abs(np.dot(approach, surface_normal)))
        alignment_score = float(np.clip(cos_angle, 0.0, 1.0))

        # ---- 4. Contact symmetry score ------------------------------------ #
        symmetry_score = _contact_symmetry(pos, g.orientation, g.gripper_width, neighbours)

        # ---- 5. Combined stability --------------------------------------- #
        g.stability_score = float(np.clip(alignment_score * symmetry_score, 0.0, 1.0))

    scores = np.array([g.stability_score for g in grasps])
    print(
        f"[Step 7] Stability estimated -- "
        f"mean={scores.mean():.3f}, max={scores.max():.3f}, "
        f"min={scores.min():.3f}"
    )
    return grasps


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _fit_plane_normal(pts: np.ndarray) -> np.ndarray:
    """
    Fit a plane to points via PCA and return the unit normal vector.
    Normal points toward the direction of least variance (smallest eigenvalue).
    """
    if pts.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0])      # default up

    centred = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    normal   = Vt[-1]                          # row corresponding to smallest sigma
    return normal / (np.linalg.norm(normal) + 1e-9)


def _contact_symmetry(
    pos:           np.ndarray,
    orientation:   np.ndarray,
    gripper_width: float,
    neighbours:    np.ndarray,
) -> float:
    """
    Measure how symmetric the point density is around both contact points.

    Contact points are at ± gripper_width/2 along the side axis (R[:,0]).
    Symmetric density -> score near 1.0.
    """
    R        = _rpy_to_rotation_matrix(orientation)
    side     = R[:, 0].astype(np.float64)
    half_w   = gripper_width / 2.0

    left_pt  = pos + half_w * side
    right_pt = pos - half_w * side

    search_r = gripper_width * 0.3 + 0.01

    # Count points near each contact location
    def count_near(centre, pts, r):
        dist = np.linalg.norm(pts - centre, axis=1)
        return int((dist < r).sum())

    n_left  = count_near(left_pt,  neighbours, search_r)
    n_right = count_near(right_pt, neighbours, search_r)

    total = n_left + n_right
    if total == 0:
        return 0.5                      # no evidence either way

    # Symmetry: 1 - abs(imbalance)
    imbalance = abs(n_left - n_right) / (total + 1e-9)
    return float(np.clip(1.0 - imbalance, 0.0, 1.0))


def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) -> 3x3 rotation matrix (Z-Y-X convention)."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr  = np.cos(r), np.sin(r)
    cp, sp  = np.cos(p), np.sin(p)
    cy, sy  = np.cos(y), np.sin(y)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return (Rz @ Ry @ Rx).astype(np.float32)
