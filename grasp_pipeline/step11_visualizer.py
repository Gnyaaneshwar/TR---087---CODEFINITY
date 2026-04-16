"""
step11_visualizer.py
=====================
STEP 11 -- VISUALIZATION

Responsibilities:
  1. Render the filtered 3-D point cloud (coloured by original RGB).
  2. Overlay grasp frames as coordinate axes at each grasp position.
  3. Colour coding:
       - Best grasp (rank 1)  -> GREEN
       - Collision grasps     -> RED
       - Other valid grasps   -> YELLOW / ORANGE
  4. Save screenshot to outputs/visualization_<timestamp>.png
  5. Optionally open interactive viewer if Open3D is available.

Fallback: if Open3D is not available, produce a matplotlib 3-D scatter plot.

Returns
-------
vis_path : str  -- path to saved visualization PNG
"""

from __future__ import annotations
import os
import warnings
import numpy as np
from datetime import datetime
from typing import List

from grasp_pipeline.utils.grasp_types import Grasp

# ---------------- COLORS ----------------
COLOUR_BEST = np.array([0.1, 0.85, 0.3])
COLOUR_COLLISION = np.array([0.95, 0.15, 0.15])
COLOUR_VALID = np.array([1.0, 0.75, 0.1])

AXIS_LEN = 0.04
SPHERE_R = 0.008

# ---------------------------------------

def visualize(points, colors, top_grasps: List[Grasp], output_dir="outputs", show=False):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_path = os.path.join(output_dir, f"visualization_{timestamp}.png")

    # ONLY TOP 3 GRASPS
    top_grasps = sorted(top_grasps, key=lambda g: g.final_score, reverse=True)[:3]

    try:
        import open3d as o3d
        return _render_open3d(points, colors, top_grasps, vis_path, show)
    except ImportError:
        warnings.warn("Open3D not installed, using matplotlib fallback")
        return _render_matplotlib(points, colors, top_grasps, vis_path)


# ---------------------------------------
#  OPEN3D RENDER
# ---------------------------------------

def _render_open3d(points, colors, grasps, out_path, show):
    import open3d as o3d

    geometries = []

    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) * 0.7)
    geometries.append(pcd)

    for i, g in enumerate(grasps):
        colour = _grasp_colour(g, i)
        R = _rpy_to_rotation_matrix(g.orientation)

        # Sphere at grasp center
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_R)
        sphere.translate(g.position.astype(np.float64))
        sphere.paint_uniform_color(colour.tolist())
        sphere.compute_vertex_normals()
        geometries.append(sphere)

        # Gripper
        gripper = _make_gripper_lines(g, R, colour)
        geometries.append(gripper)

    # Render
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)

    for g in geometries:
        vis.add_geometry(g)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.0, -0.5, 1.0])
    ctr.set_up([0.0, -1.0, 0.2])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path, do_render=True)

    if show:
        vis.run()

    vis.destroy_window()
    return out_path


# ---------------------------------------
# GRIPPER (FINAL VERSION)
# ---------------------------------------

def _make_gripper_lines(g: Grasp, R: np.ndarray, colour: np.ndarray):
    import open3d as o3d

    side = R[:, 0]
    approach = R[:, 2]

    half_w = max(g.gripper_width / 2.0, 0.02)
    depth = 0.08  # extended for visibility

    left_base = g.position + half_w * side
    right_base = g.position - half_w * side

    left_tip = left_base + depth * approach
    right_tip = right_base + depth * approach

    # Slight offset to simulate thickness
    offset = 0.002 * side

    left_base2 = left_base + offset
    right_base2 = right_base + offset
    left_tip2 = left_tip + offset
    right_tip2 = right_tip + offset

    pts = o3d.utility.Vector3dVector([
        left_base, left_tip, right_base, right_tip,
        left_base2, left_tip2, right_base2, right_tip2
    ])

    lns = o3d.utility.Vector2iVector([
        [0,1], [2,3], [0,2],      # main
        [4,5], [6,7], [4,6]       # thickness layer
    ])

    colors = o3d.utility.Vector3dVector([colour.tolist()] * len(lns))

    line_set = o3d.geometry.LineSet()
    line_set.points = pts
    line_set.lines = lns
    line_set.colors = colors

    return line_set


# ---------------------------------------
# -- COLOR LOGIC --
# ---------------------------------------

def _grasp_colour(g: Grasp, idx):
    if idx == 0:
        return COLOUR_BEST
    if g.collision_status == "collision":
        return COLOUR_COLLISION
    return COLOUR_VALID


# -- ROTATION HELPER (used by both Open3D and Matplotlib paths) --

def _rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) -> 3x3 rotation matrix, Z-Y-X convention."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [0,  0,  1]], dtype=np.float64)
    Ry = np.array([[cp,  0, sp], [0,   1,  0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1,   0,  0], [0,  cr, -sr], [0, sr,  cr]], dtype=np.float64)
    return (Rz @ Ry @ Rx).astype(np.float32)


# -- MATPLOTLIB FALLBACK (used when Open3D is not installed) --

def _render_matplotlib(
    points:     np.ndarray,
    colors:     np.ndarray,
    top_grasps: List[Grasp],
    out_path:   str,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    fig = plt.figure(figsize=(14, 10), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")

    # Subsample point cloud for performance
    N = points.shape[0]
    max_pts = 5000
    if N > max_pts:
        idx = np.random.choice(N, max_pts, replace=False)
        pts_plot = points[idx]
        col_plot = colors[idx]
    else:
        pts_plot, col_plot = points, colors

    col_plot_dark = np.clip(col_plot * 0.7, 0, 1)
    ax.scatter(
        pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
        c=col_plot_dark, s=1.0, alpha=0.5, depthshade=True
    )

    # Grasp markers + gripper lines
    for i, g in enumerate(top_grasps):
        colour = _grasp_colour(g, i)
        mpl_c  = colour.tolist()
        p      = g.position.astype(np.float64)
        sz     = 160 if i == 0 else 80

        ax.scatter([p[0]], [p[1]], [p[2]], c=[mpl_c], s=sz,
                   edgecolors="white", linewidths=0.6, zorder=5)

        R = _rpy_to_rotation_matrix(g.orientation).astype(np.float64)
        side     = R[:, 0]
        approach = R[:, 2]
        half_w   = max(float(g.gripper_width) / 2.0, 0.02)
        depth    = 0.08

        # Gripper fingers
        for sign in (+1.0, -1.0):
            base = p + sign * half_w * side
            tip  = base + depth * approach
            ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
                    color=mpl_c, linewidth=2.0, alpha=0.9)

        # Crossbar between finger bases
        lb = p + half_w * side
        rb = p - half_w * side
        ax.plot([lb[0], rb[0]], [lb[1], rb[1]], [lb[2], rb[2]],
                color=mpl_c, linewidth=1.5, alpha=0.8)

        # Score label
        label = f"#{i+1} {g.final_score:.2f}"
        ax.text(p[0], p[1], p[2] + 0.02, label,
                color="white", fontsize=8, ha="center", fontweight="bold")

    # Legend
    legend = [
        mpatches.Patch(color=[0.1, 0.85, 0.3],   label="Best grasp"),
        mpatches.Patch(color=[1.0, 0.75, 0.1],   label="Valid grasp"),
        mpatches.Patch(color=[0.95, 0.15, 0.15], label="Collision grasp"),
    ]
    ax.legend(handles=legend, loc="upper left",
              facecolor="#2a2a4e", edgecolor="white",
              labelcolor="white", fontsize=9)

    ax.set_xlabel("X (m)", color="white", fontsize=9)
    ax.set_ylabel("Y (m)", color="white", fontsize=9)
    ax.set_zlabel("Z (m)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_title("Robotic Grasp Planning -- Top Grasps",
                 color="white", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"[Step 11] Visualization saved -> {os.path.abspath(out_path)}")
    return out_path


# -------------------------------