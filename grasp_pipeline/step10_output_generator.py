"""
step10_output_generator.py
===========================
STEP 10 -- OUTPUT GENERATION

Responsibilities:
  1. Serialise top-10 grasps to a structured JSON document.
  2. Include metadata: timestamp, total candidates, pipeline config.
  3. Save to outputs/grasps_<timestamp>.json
  4. Return the output file path and the dict.

JSON schema per grasp:
  {
    "rank":               int,
    "position":           {"x": float, "y": float, "z": float},
    "orientation":        {"roll": float, "pitch": float, "yaw": float},
    "gripper_width":      float,
    "confidence":         float,
    "stability_score":    float,
    "collision_free_score": float,
    "final_score":        float,
    "collision_status":   str,
    "physics_validated":  bool,
    "physics_score":      float
  }

Returns
-------
output_path : str  -- absolute path to saved JSON file
result_dict : dict -- full serialised result
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import List, Tuple, Optional

from grasp_pipeline.utils.grasp_types import Grasp


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate_output(
    top_grasps:       List[Grasp],
    total_candidates: int,
    output_dir:       str = "outputs",
    extra_metadata:   Optional[dict] = None,
) -> Tuple[str, dict]:
    """
    Serialise top grasps to JSON and save to disk.

    Parameters
    ----------
    top_grasps       : List[Grasp]  -- ranked, validated grasps
    total_candidates : int          -- total before filtering
    output_dir       : str          -- directory to save JSON
    extra_metadata   : dict | None  -- optional extra fields in the header

    Returns
    -------
    output_path : str   -- where the JSON was saved
    result_dict : dict  -- Python dict (same content as JSON)
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"grasps_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    # ---- Build result dict ----------------------------------------------- #
    result_dict: dict = {
        "metadata": {
            "timestamp":        datetime.now().isoformat(),
            "total_candidates": total_candidates,
            "top_k_returned":   len(top_grasps),
            "scoring_weights": {
                "confidence":         0.4,
                "stability_score":    0.3,
                "collision_free":     0.3,
            },
        },
        "top_grasps": [g.to_dict() for g in top_grasps],
    }

    if extra_metadata:
        result_dict["metadata"].update(extra_metadata)

    # ---- Write JSON ------------------------------------------------------- #
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, indent=2)

    print(f"[Step 10] Output saved -> {os.path.abspath(output_path)}")
    _pretty_print_summary(top_grasps)

    return output_path, result_dict


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _pretty_print_summary(grasps: List[Grasp]) -> None:
    """Print a formatted table of the top grasps to stdout."""
    header = (
        f"{'Rank':>4}  {'X':>7}  {'Y':>7}  {'Z':>7}  "
        f"{'Conf':>5}  {'Stab':>5}  {'Coll':>5}  "
        f"{'Final':>6}  {'Phys':>5}  {'Valid':>5}"
    )
    sep = "-" * len(header)
    print(f"\n[Step 10] === Top {len(grasps)} Grasps ===")
    print(header)
    print(sep)
    for g in grasps:
        p = g.position
        print(
            f"{g.rank:>4}  {p[0]:>7.3f}  {p[1]:>7.3f}  {p[2]:>7.3f}  "
            f"{g.confidence:>5.3f}  {g.stability_score:>5.3f}  "
            f"{g.collision_free_score:>5.3f}  {g.final_score:>6.4f}  "
            f"{g.physics_score:>5.3f}  {'OK' if g.physics_validated else 'N':>5}"
        )
    print(sep)
