"""
step05_initial_scorer.py
========================
STEP 5 -- INITIAL SCORING

Responsibilities:
  1. For each grasp candidate, assign a confidence score ∈ [0, 1].
  2. Score is based on three geometric heuristics:
       a. Point density  -- denser neighbourhood -> more stable contact points.
       b. Normal consistency -- low variance in k-NN normals -> cleaner surface.
       c. Centroid proximity -- grasps near the object centroid preferred.
  3. Normalise all scores to [0, 1].
  4. Write `grasp.confidence` for every Grasp in the list.

Returns
-------
List[Grasp]  -- same list with `confidence` field populated
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from typing import List

from grasp_pipeline.utils.grasp_types import Grasp


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def score_initial(
    grasps:        List[Grasp],
    points:        np.ndarray,
    radius:        float = 0.05,    # neighbourhood radius in metres
    k_normal:      int   = 20,      # k for normal-consistency check
) -> List[Grasp]:
    """
    Assign initial confidence scores to all grasp candidates.

    Parameters
    ----------
    grasps  : List[Grasp]  -- grasp candidates (orientation already set)
    points  : (M, 3) float32 -- foreground point cloud
    radius  : float          -- sphere radius for density count
    k_normal: int            -- k-NN for normal fit variance

    Returns
    -------
    List[Grasp]  -- grasps with `confidence` populated in [0, 1]
    """
    if len(grasps) == 0:
        return grasps

    tree     = cKDTree(points)
    centroid = points.mean(axis=0)
    max_dist = np.linalg.norm(points - centroid, axis=1).max() + 1e-9

    # ---------------------------------------------------------------------- #
    # Compute raw sub-scores for every grasp                                  #
    # ---------------------------------------------------------------------- #
    density_scores    = np.zeros(len(grasps), dtype=np.float64)
    consistency_scores= np.zeros(len(grasps), dtype=np.float64)
    centroid_scores   = np.zeros(len(grasps), dtype=np.float64)

    for i, g in enumerate(grasps):
        pos = g.position.astype(np.float64)

        # ---- a. Point density -------------------------------------------- #
        idx_ball = tree.query_ball_point(pos, r=radius)
        density_scores[i] = len(idx_ball)

        # ---- b. Normal consistency ---------------------------------------- #
        if len(idx_ball) >= k_normal:
            neighbours = points[idx_ball[:k_normal]]
        elif len(idx_ball) >= 3:
            neighbours = points[idx_ball]
        else:
            _, nn_idx  = tree.query(pos, k=max(3, k_normal))
            neighbours = points[nn_idx]

        consistency_scores[i] = _normal_consistency(neighbours)

        # ---- c. Centroid proximity ---------------------------------------- #
        dist = np.linalg.norm(pos - centroid)
        centroid_scores[i] = 1.0 - dist / max_dist

    # ---------------------------------------------------------------------- #
    # Min-max normalise each sub-score to [0, 1]                              #
    # ---------------------------------------------------------------------- #
    density_scores     = _minmax(density_scores)
    consistency_scores = _minmax(consistency_scores)
    centroid_scores    = _minmax(centroid_scores)

    # Weighted combination  (density 40%, consistency 40%, centroid 20%)
    combined = (
        0.40 * density_scores
        + 0.40 * consistency_scores
        + 0.20 * centroid_scores
    )
    combined = _minmax(combined)    # re-normalise final blend

    for i, g in enumerate(grasps):
        g.confidence = float(np.clip(combined[i], 0.0, 1.0))

    scores = combined
    print(
        f"[Step 5] Confidence assigned -- "
        f"mean={scores.mean():.3f}, max={scores.max():.3f}, "
        f"min={scores.min():.3f}"
    )
    return grasps


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normal_consistency(pts: np.ndarray) -> float:
    """
    Measure how well the local neighbourhood fits a plane (PCA planarity).
    Returns 1.0 for a perfect flat surface, 0.0 for an isotropic cloud.
    """
    if pts.shape[0] < 3:
        return 0.5
    cov  = np.cov(pts.T)
    vals = np.linalg.eigvalsh(cov)    # ascending
    total = vals.sum() + 1e-12
    # Planarity: small third eigenvalue relative to the others
    planarity = 1.0 - (vals[0] / total)
    return float(np.clip(planarity, 0.0, 1.0))


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1].  Returns uniform 0.5 if all equal."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - lo) / (hi - lo)
