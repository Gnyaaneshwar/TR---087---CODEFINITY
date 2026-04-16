"""
step06_collision_checker.py
============================
STEP 6 -- COLLISION CHECKING

Responsibilities:
  1. Model gripper as two rectangular finger boxes (left + right finger).
  2. For each grasp, transform finger boxes into world frame using grasp pose.
  3. Test whether any point-cloud points occupy the finger volumes.
  4. If collision -> heavy penalty: collision_free_score = 0.0, status = "collision"
     If clear    -> collision_free_score = 1.0,                  status = "clear"

Gripper geometry (default: parallel-jaw, e.g. Robotiq 2F-85):
  - Finger depth  (along approach axis) = 0.06 m
  - Finger height (along grasp axis)    = 0.02 m
  - Finger offset from centre (+-width/2) along side axis

Returns
-------
List[Grasp]  -- grasps with `collision_free_score` and `collision_status` set
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from typing import List

from grasp_pipeline.utils.grasp_types import Grasp


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def check_collisions(
    grasps:           List[Grasp],
    points:           np.ndarray,
    finger_depth:     float = 0.06,   # finger reach along approach axis (m)
    finger_thickness: float = 0.01,   # solid body thickness of each finger (m)
    palm_depth:       float = 0.02,   # palm zone depth above approach point (m)
    clearance:        float = 0.002,  # safety margin (m)
) -> List[Grasp]:
    """
    Check every grasp for collisions with the point cloud.

    Collision model (parallel-jaw gripper):
      - TWO FINGER BODIES: solid rectangular boxes flanking the jaw opening.
        Each finger is centred at +/- (half_width + finger_thickness/2) from
        the grasp centre along the side axis.
      - PALM ZONE: flat box above the grasp point spanning full gripper width.

    A grasp is collision-free when NO scene points fall inside either finger
    body or the palm zone.

    Parameters
    ----------
    grasps            : List[Grasp]
    points            : (M, 3) float32 -- foreground points in metres
    finger_depth      : float
    finger_thickness  : float
    palm_depth        : float
    clearance         : float

    Returns
    -------
    List[Grasp]  -- collision_free_score and collision_status populated
    """
    tree = cKDTree(points)
    n_clear = n_collision = 0

    for g in grasps:
        R        = _rpy_to_rotation_matrix(g.orientation)
        approach = R[:, 2].astype(np.float64)
        side     = R[:, 0].astype(np.float64)
        pos      = g.position.astype(np.float64)
        half_w   = float(g.gripper_width) / 2.0

        collision = False

        # ---- Left and right finger bodies ---------------------------------
        for sign in (+1.0, -1.0):
            # Finger centre: offset outward past the jaw opening + half thickness
            fc = (pos
                  + sign * (half_w + finger_thickness / 2.0 + clearance) * side
                  + (finger_depth / 2.0) * approach)
            if _obb_has_points(tree, points, fc, R,
                               hx=finger_thickness / 2.0 + clearance,
                               hy=finger_depth     / 2.0 + clearance,
                               hz=finger_depth     / 2.0 + clearance):
                collision = True
                break

        # ---- Palm zone (base of gripper, above approach point) ------------
        if not collision:
            palm_c = pos + (finger_depth + palm_depth / 2.0) * approach
            if _obb_has_points(tree, points, palm_c, R,
                               hx=half_w + finger_thickness + clearance,
                               hy=palm_depth / 2.0 + clearance,
                               hz=palm_depth / 2.0 + clearance):
                collision = True

        if collision:
            g.collision_free_score = 0.0
            g.collision_status     = "collision"
            n_collision += 1
        else:
            g.collision_free_score = 1.0
            g.collision_status     = "clear"
            n_clear += 1

    total = max(len(grasps), 1)
    print(
        f"[Step 6] Collision check -- "
        f"{n_clear} clear, {n_collision} collisions "
        f"({100 * n_clear / total:.1f}% pass rate)"
    )
    return grasps


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) -> 3x3 rotation matrix (Z-Y-X convention)."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr  = np.cos(r), np.sin(r)
    cp, sp  = np.cos(p), np.sin(p)
    cy, sy  = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [0,   0,  1]], dtype=np.float64)
    Ry = np.array([[cp,  0, sp], [0,   1,  0], [-sp, 0,  cp]], dtype=np.float64)
    Rx = np.array([[1,   0,  0], [0,  cr, -sr], [0,  sr,  cr]], dtype=np.float64)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _obb_has_points(
    tree:   cKDTree,
    points: np.ndarray,
    centre: np.ndarray,
    R:      np.ndarray,
    hx:     float,
    hy:     float,
    hz:     float,
) -> bool:
    """
    Return True if any point in `points` lies inside the oriented bounding box
    (centre, rotation R, half-extents hx/hy/hz).

    Fast path: bounding-sphere pre-filter via KD-tree, then exact OBB test.
    """
    bounding_r     = float(np.sqrt(hx**2 + hy**2 + hz**2))
    candidate_idx  = tree.query_ball_point(centre, r=bounding_r)
    if not candidate_idx:
        return False

    cands = points[candidate_idx].astype(np.float64)
    local = (cands - centre) @ R.astype(np.float64)   # transform to OBB frame

    inside = (
        (np.abs(local[:, 0]) <= hx) &
        (np.abs(local[:, 1]) <= hy) &
        (np.abs(local[:, 2]) <= hz)
    )
    return bool(inside.any())
