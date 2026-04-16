"""
step04_grasp_candidates.py
===========================
STEP 4 -- GRASP CANDIDATE GENERATION

Responsibilities:
  1. Sample surface points from the filtered point cloud.
  2. Estimate local surface normal at each sample via PCA on k-neighbours.
  3. Generate approach vectors perpendicular to surface normal.
  4. Sample roll angles around the approach axis.
  5. Produce at minimum N=20 grasp candidates (default N=50).

Each grasp has:
  - position    (x, y, z)
  - orientation (roll, pitch, yaw)  derived from approach + roll
  - gripper_width -- estimated from local point spread

Note
----
This is an analytical grasp generator that mirrors the output schema of
GraspNet / Contact-GraspNet.  To swap in a real model, replace the body
of `generate_grasp_candidates()` and keep the return signature identical.

Returns
-------
List[Grasp]  -- N candidate grasps with position, orientation, gripper_width
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from typing import List
import warnings

from grasp_pipeline.utils.grasp_types import Grasp


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate_grasp_candidates(
    points:         np.ndarray,
    n_candidates:   int   = 50,      # must be >= 20 per spec
    k_neighbours:   int   = 30,      # k-NN for normal estimation
    n_roll_samples: int   = 4,       # roll angles per surface sample
    gripper_min:    float = 0.04,    # min gripper width (m)
    gripper_max:    float = 0.12,    # max gripper width (m)
    seed:           int   = 42,
) -> List[Grasp]:
    """
    Generate analytical grasp candidates from a filtered point cloud.

    Parameters
    ----------
    points       : (M, 3) float32 -- foreground point cloud
    n_candidates : int            -- total grasps to generate (>= 20)
    k_neighbours : int            -- k for local normal estimation
    n_roll_samples: int           -- roll angles per anchor point
    gripper_min/max: float        -- gripper width bounds
    seed         : int            -- random seed for reproducibility

    Returns
    -------
    List[Grasp] -- at least n_candidates grasps
    """
    n_candidates = max(n_candidates, 20)   # spec: minimum 20
    rng = np.random.default_rng(seed)

    M = points.shape[0]
    if M < k_neighbours:
        raise ValueError(
            f"[Step 4] Too few foreground points ({M}) for "
            f"normal estimation (need >= {k_neighbours})."
        )

    # --- Build KD-tree once for all queries -------------------------------- #
    tree = cKDTree(points)

    # --- Sample anchor points --------------------------------------------- #
    n_anchors = max(n_candidates // n_roll_samples, 5)
    anchor_idx = rng.choice(M, size=n_anchors, replace=(n_anchors > M))
    anchors    = points[anchor_idx]                # (A, 3)

    # --- Estimate surface normals via local PCA ---------------------------- #
    normals, widths = _estimate_normals_and_widths(
        points, anchors, tree, k_neighbours, gripper_min, gripper_max
    )

    # --- Generate grasps for each anchor ----------------------------------- #
    grasps: List[Grasp] = []
    roll_angles = np.linspace(0, np.pi, n_roll_samples, endpoint=False)

    for i, (anchor, normal, width) in enumerate(zip(anchors, normals, widths)):
        for roll in roll_angles:
            approach = -normal              # approach from above the surface
            rpy      = _approach_to_rpy(approach, roll)

            g = Grasp(
                position      = anchor.copy(),
                orientation   = rpy,
                gripper_width = float(width),
            )
            grasps.append(g)

    # --- Pad to n_candidates if we ended up with fewer -------------------- #
    while len(grasps) < n_candidates:
        i   = rng.integers(len(grasps))
        src = grasps[i]
        jitter = rng.uniform(-0.01, 0.01, size=3).astype(np.float32)
        g = Grasp(
            position      = src.position + jitter,
            orientation   = src.orientation + rng.uniform(-0.1, 0.1, size=3),
            gripper_width = float(np.clip(
                src.gripper_width + rng.uniform(-0.01, 0.01),
                gripper_min, gripper_max
            )),
        )
        grasps.append(g)

    # Keep exactly n_candidates (trim if we generated too many)
    grasps = grasps[:n_candidates]

    print(
        f"[Step 4] Generated {len(grasps)} grasp candidates "
        f"from {n_anchors} surface anchors x {n_roll_samples} roll angles"
    )
    return grasps


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _estimate_normals_and_widths(
    all_points: np.ndarray,
    anchors:    np.ndarray,
    tree:       cKDTree,
    k:          int,
    w_min:      float,
    w_max:      float,
) -> tuple:
    """
    For each anchor, query k-NN, fit PCA, return normal and width estimate.
    Normal is oriented to point away from centroid (upward convention).
    """
    normals = np.zeros((len(anchors), 3), dtype=np.float32)
    widths  = np.zeros(len(anchors),    dtype=np.float32)
    centroid = all_points.mean(axis=0)

    for i, anchor in enumerate(anchors):
        _, idx = tree.query(anchor, k=k)
        neighbourhood = all_points[idx]        # (k, 3)

        # PCA: smallest eigenvalue -> surface normal direction
        cov  = np.cov(neighbourhood.T)         # (3, 3)
        vals, vecs = np.linalg.eigh(cov)       # ascending eigenvalues
        normal = vecs[:, 0]                    # eigenvector for smallest lambda

        # Orient normal away from centroid
        if np.dot(normal, anchor - centroid) < 0:
            normal = -normal

        normals[i] = normal.astype(np.float32)

        # Gripper width from neighbourhood spread perpendicular to normal
        proj = neighbourhood - (neighbourhood @ normal)[:, None] * normal
        spread = proj.max(axis=0) - proj.min(axis=0)
        width  = float(np.clip(np.linalg.norm(spread[:2]) * 0.6, w_min, w_max))
        widths[i] = width

    return normals, widths


def _approach_to_rpy(approach: np.ndarray, roll: float) -> np.ndarray:
    """
    Convert an approach direction vector + roll angle to (roll, pitch, yaw).

    Convention: approach vector defines the Z-axis of the gripper frame.
    """
    approach = approach / (np.linalg.norm(approach) + 1e-9)

    # Pitch from approach Z component
    pitch = float(np.arcsin(np.clip(-approach[2], -1.0, 1.0)))

    # Yaw from XY plane projection
    yaw = float(np.arctan2(approach[1], approach[0]))

    return np.array([roll, pitch, yaw], dtype=np.float32)
