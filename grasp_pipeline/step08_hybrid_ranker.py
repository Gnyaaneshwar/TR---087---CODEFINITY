"""
step08_hybrid_ranker.py
========================
STEP 8 -- HYBRID RE-RANKING

Responsibilities:
  1. Compute final_score for each grasp using the weighted formula:
       final_score = 0.4 x confidence
                   + 0.3 x stability_score
                   + 0.3 x collision_free_score
  2. Sort grasps descending by final_score.
  3. Select top K grasps (default K = 10).
  4. Assign rank field (1 = best).

Returns
-------
List[Grasp]  -- top K grasps, sorted by descending final_score, with rank set
"""

from __future__ import annotations
import numpy as np
from typing import List

from grasp_pipeline.utils.grasp_types import Grasp


# Weights as specified in the problem statement
W_CONFIDENCE  = 0.4
W_STABILITY   = 0.3
W_COLLISION   = 0.3


def rank_and_select(
    grasps: List[Grasp],
    top_k:  int = 10,
) -> List[Grasp]:
    """
    Re-rank all grasps by hybrid score and return the top K.

    Parameters
    ----------
    grasps : List[Grasp]  -- all candidates with confidence, stability,
                            collision_free_score already set
    top_k  : int          -- number of top grasps to return (default 10)

    Returns
    -------
    List[Grasp]  -- top_k grasps sorted descending by final_score
    """
    if not grasps:
        return []

    # --- Compute final hybrid scores -------------------------------------- #
    for g in grasps:
        g.final_score = float(
            W_CONFIDENCE * g.confidence
            + W_STABILITY  * g.stability_score
            + W_COLLISION  * g.collision_free_score
        )

    # --- Sort descending --------------------------------------------------- #
    grasps_sorted = sorted(grasps, key=lambda g: g.final_score, reverse=True)

    # --- Select top K ------------------------------------------------------ #
    top_grasps = grasps_sorted[:top_k]

    # --- Assign rank ------------------------------------------------------- #
    for rank, g in enumerate(top_grasps, start=1):
        g.rank = rank

    scores = np.array([g.final_score for g in top_grasps])
    print(
        f"[Step 8] Re-ranking complete -- top {len(top_grasps)} selected\n"
        f"         scores: best={scores[0]:.4f}, "
        f"worst={scores[-1]:.4f}, mean={scores.mean():.4f}"
    )

    # Pretty-print the top-3
    for g in top_grasps[:3]:
        print(f"         {g}")

    return top_grasps
