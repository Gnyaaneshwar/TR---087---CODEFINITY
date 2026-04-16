"""
grasp_types.py
Shared data structures for the grasp planning pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
from typing import Optional


@dataclass
class Grasp:
    """
    Represents a single 6-DOF grasp candidate.

    Attributes
    ----------
    position        : np.ndarray (3,) -- (x, y, z) in metres
    orientation     : np.ndarray (3,) -- (roll, pitch, yaw) in radians
    gripper_width   : float            -- distance between fingers in metres
    confidence      : float ∈ [0,1]   -- model / heuristic confidence
    stability_score : float ∈ [0,1]   -- surface-alignment stability
    collision_free_score : float ∈ [0,1] -- 1 = no collision, 0 = collision
    final_score     : float ∈ [0,1]   -- weighted hybrid score
    collision_status: str             -- "clear" | "collision" | "unknown"
    physics_validated: bool           -- passed physics simulation?
    physics_score   : float ∈ [0,1]   -- simulation success metric
    rank            : int             -- final rank (1 = best)
    """
    position: np.ndarray
    orientation: np.ndarray          # (roll, pitch, yaw)
    gripper_width: float = 0.08

    confidence: float = 0.0
    stability_score: float = 0.0
    collision_free_score: float = 0.0
    final_score: float = 0.0

    collision_status: str = "unknown"
    physics_validated: bool = False
    physics_score: float = 0.0
    rank: int = 0

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return {
            "rank": self.rank,
            "position": {
                "x": float(self.position[0]),
                "y": float(self.position[1]),
                "z": float(self.position[2]),
            },
            "orientation": {
                "roll":  float(self.orientation[0]),
                "pitch": float(self.orientation[1]),
                "yaw":   float(self.orientation[2]),
            },
            "gripper_width": float(self.gripper_width),
            "confidence": float(self.confidence),
            "stability_score": float(self.stability_score),
            "collision_free_score": float(self.collision_free_score),
            "final_score": float(self.final_score),
            "collision_status": self.collision_status,
            "physics_validated": bool(self.physics_validated),
            "physics_score": float(self.physics_score),
        }

    def __repr__(self) -> str:
        p = self.position
        o = self.orientation
        return (
            f"Grasp(rank={self.rank}, pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}), "
            f"score={self.final_score:.3f}, collision={self.collision_status})"
        )
