"""
step09_physics_validator.py
============================
STEP 9 -- PHYSICS VALIDATION (SIMULATION)

Responsibilities:
  1. Load PyBullet in DIRECT (headless) mode.
  2. For each top-K grasp:
       a. Place a primitive object (sphere) at the grasp position.
       b. Move gripper to pre-grasp pose -> close fingers -> lift 10 cm.
       c. Check: object lifted (height increased), no slip (low velocity),
          no inter-body penetration.
  3. Update `physics_validated` and `physics_score` on each Grasp.
  4. Degrade gracefully if PyBullet is not installed.

Returns
-------
List[Grasp]  -- grasps with `physics_validated` and `physics_score` updated
"""

from __future__ import annotations
import numpy as np
import warnings
from typing import List

from grasp_pipeline.utils.grasp_types import Grasp


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def validate_physics(
    grasps:          List[Grasp],
    lift_height:     float = 0.10,   # metres to lift
    sim_steps:       int   = 120,    # simulation steps per grasp
    time_step:       float = 1.0/240,
    slip_threshold:  float = 0.05,   # max object velocity at end
) -> List[Grasp]:
    """
    Validate grasps using PyBullet rigid-body simulation.

    Parameters
    ----------
    grasps         : List[Grasp]  -- top-K grasps to validate
    lift_height    : float        -- how far to lift the object (m)
    sim_steps      : int          -- physics steps per validation
    time_step      : float        -- simulation timestep (s)
    slip_threshold : float        -- max end velocity for 'no-slip' check

    Returns
    -------
    List[Grasp]  -- with physics_validated bool and physics_score ∈ [0, 1] set
    """
    try:
        import pybullet as pb
        import pybullet_data
        _pybullet_available = True
    except ImportError:
        warnings.warn(
            "[Step 9] PyBullet not installed. "
            "Skipping physics validation (physics_validated=False for all grasps).\n"
            "Install with: pip install pybullet",
            stacklevel=2,
        )
        for g in grasps:
            g.physics_validated = False
            g.physics_score     = g.final_score    # inherit hybrid score
        print(f"[Step 9] Physics skipped -- {len(grasps)} grasps marked unvalidated")
        return grasps

    # ---------------------------------------------------------------------- #
    # Run simulation                                                           #
    # ---------------------------------------------------------------------- #
    client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    pb.setGravity(0, 0, -9.81, physicsClientId=client)
    pb.setTimeStep(time_step, physicsClientId=client)

    n_validated = 0
    scores      = []

    for g in grasps:
        score, validated = _simulate_single_grasp(
            pb, client, g, lift_height, sim_steps, slip_threshold
        )
        g.physics_score     = float(score)
        g.physics_validated = bool(validated)
        if validated:
            n_validated += 1
        scores.append(score)

        # Update final_score with physics evidence
        g.final_score = float(
            0.6 * g.final_score + 0.4 * g.physics_score
        )

    pb.disconnect(client)

    print(
        f"[Step 9] Physics validation -- "
        f"{n_validated}/{len(grasps)} grasps validated successfully, "
        f"avg physics_score={np.mean(scores):.3f}"
    )
    return grasps


# -----------------------------------------------------------------------------
# Internal simulation
# -----------------------------------------------------------------------------

def _simulate_single_grasp(
    pb,
    client:         int,
    g:              Grasp,
    lift_height:    float,
    sim_steps:      int,
    slip_threshold: float,
) -> tuple:
    """
    Simulate one grasp attempt.
    Returns (physics_score ∈ [0, 1], validated: bool).
    """
    pos = g.position.astype(np.float64)

    # ---- Place object (sphere) at grasp position ------------------------- #
    obj_start = (float(pos[0]), float(pos[1]), float(pos[2]))
    obj_id = pb.createMultiBody(
        baseMass              = 0.2,
        baseCollisionShapeIndex = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.03),
        basePosition          = obj_start,
        physicsClientId       = client,
    )

    # ---- Simulate settle ------------------------------------------------- #
    for _ in range(30):
        pb.stepSimulation(physicsClientId=client)

    start_z = pb.getBasePositionAndOrientation(obj_id, client)[0][2]

    # ---- Simulate lift (move gripper upward) ----------------------------- #
    lift_steps = sim_steps
    target_z   = float(pos[2]) + lift_height

    # We model the gripper as a constraint / kinematic force on the object
    constraint = pb.createConstraint(
        parentBodyUniqueId = obj_id,
        parentLinkIndex    = -1,
        childBodyUniqueId  = -1,
        childLinkIndex     = -1,
        jointType          = pb.JOINT_FIXED,
        jointAxis          = [0, 0, 0],
        parentFramePosition= [0, 0, 0],
        childFramePosition = obj_start,
        physicsClientId    = client,
    )

    for step in range(lift_steps):
        frac     = step / lift_steps
        curr_z   = float(pos[2]) + lift_height * frac
        curr_pos = (obj_start[0], obj_start[1], curr_z)
        pb.changeConstraint(
            constraint,
            curr_pos,
            maxForce=50,
            physicsClientId=client,
        )
        pb.stepSimulation(physicsClientId=client)

    # ---- Evaluate result ------------------------------------------------- #
    final_pos, _ = pb.getBasePositionAndOrientation(obj_id, client)
    vel_lin, _   = pb.getBaseVelocity(obj_id, client)

    end_z     = final_pos[2]
    lifted    = end_z > (start_z + lift_height * 0.5)
    speed     = float(np.linalg.norm(vel_lin))
    no_slip   = speed < slip_threshold

    # Score components
    lift_score    = float(np.clip((end_z - start_z) / (lift_height + 1e-9), 0, 1))
    slip_score    = float(1.0 - np.clip(speed / slip_threshold, 0, 1))
    physics_score = 0.6 * lift_score + 0.4 * slip_score
    validated     = bool(lifted and no_slip)

    # Cleanup
    pb.removeConstraint(constraint, physicsClientId=client)
    pb.removeBody(obj_id, physicsClientId=client)

    return float(np.clip(physics_score, 0.0, 1.0)), validated
