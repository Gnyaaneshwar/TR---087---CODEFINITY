"""
step03_segmentation.py
=======================
STEP 3 -- SCENE SEGMENTATION

Responsibilities:
  1. Separate foreground objects from background in the point cloud.
  2. Strategy:
       a. Depth-threshold based floor/wall removal.
       b. DBSCAN clustering to isolate discrete object clusters.
       c. Keep all clusters that are NOT the dominant flat background.
  3. Return the filtered point cloud (foreground only).

Returns
-------
P_filtered        : np.ndarray (M, 3) float32 -- foreground 3-D points
C_filtered        : np.ndarray (M, 3) float32 -- corresponding colours
foreground_mask   : np.ndarray (N,)   bool     -- mask over input points
"""

from __future__ import annotations
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from typing import Tuple
import warnings


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def segment_scene(
    points:          np.ndarray,
    colors:          np.ndarray,
    depth_threshold: float = 0.85,   # normalised depth -- pixels deeper than this are background
    dbscan_eps:      float = 0.02,   # DBSCAN neighbourhood radius in metres
    dbscan_min:      int   = 30,     # DBSCAN minimum points per cluster
    max_points:      int   = 20_000, # subsample for speed if larger
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment foreground objects from background using depth thresholding
    and DBSCAN clustering.

    Parameters
    ----------
    points          : (N, 3) float32 -- full point cloud
    colors          : (N, 3) float32 -- corresponding colours
    depth_threshold : float           -- Z-axis percentile cutoff for background
    dbscan_eps      : float           -- cluster radius in metres
    dbscan_min      : int             -- min cluster size
    max_points      : int             -- max points before subsampling

    Returns
    -------
    P_filtered      : (M, 3) float32  -- foreground points
    C_filtered      : (M, 3) float32  -- foreground colours
    fg_mask         : (N,)   bool     -- which input points are foreground
    """
    N = points.shape[0]

    # --- Subsample for speed ----------------------------------------------- #
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        pts_sub  = points[idx]
        col_sub  = colors[idx]
    else:
        idx      = np.arange(N)
        pts_sub  = points
        col_sub  = colors

    # --- Step 3a: Depth threshold to remove far background ----------------- #
    Z = pts_sub[:, 2]
    z_min, z_max = Z.min(), Z.max()
    z_thresh = z_min + depth_threshold * (z_max - z_min)
    near_mask = Z <= z_thresh

    pts_near = pts_sub[near_mask]
    col_near = col_sub[near_mask]

    if pts_near.shape[0] < dbscan_min:
        warnings.warn(
            "[Step 3] Too few points after depth threshold. "
            "Returning all points as foreground.", stacklevel=2
        )
        P_filtered = pts_sub
        C_filtered = col_sub
        fg_mask    = np.ones(N, dtype=bool)
        print(f"[Step 3] Fallback: {P_filtered.shape[0]:,} foreground points")
        return P_filtered, C_filtered, fg_mask

    # --- Step 3b: DBSCAN clustering ---------------------------------------- #
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min, algorithm="ball_tree", n_jobs=-1)
    labels = db.fit_predict(pts_near[:, :3])   # cluster on XYZ only

    unique_labels = set(labels)
    unique_labels.discard(-1)                  # -1 = noise in DBSCAN

    if len(unique_labels) == 0:
        # No clusters found -- keep all near points
        warnings.warn(
            "[Step 3] DBSCAN found no clusters. "
            "Falling back to depth-threshold only.", stacklevel=2
        )
        fg_mask_sub = near_mask
        P_filtered  = pts_near
        C_filtered  = col_near
    else:
        # --- Step 3c: Remove dominant flat background cluster -------------- #
        # The background cluster is the one whose points lie most in a plane.
        # Measure planarity = (lambda2 - lambda3) / lambda1  via PCA eigenvalues.
        bg_label = _find_background_cluster(pts_near, labels, unique_labels)

        fg_labels_mask = (labels != -1) & (labels != bg_label)
        fg_sub_mask    = np.zeros(pts_near.shape[0], dtype=bool)
        fg_sub_mask[fg_labels_mask] = True     # foreground within near cloud

        # Back-map into full near-cloud boolean
        fg_mask_sub = np.zeros(pts_sub.shape[0], dtype=bool)
        near_indices = np.where(near_mask)[0]
        fg_mask_sub[near_indices[fg_sub_mask]] = True

        P_filtered = pts_sub[fg_mask_sub]
        C_filtered = col_sub[fg_mask_sub]

        if P_filtered.shape[0] < 10:
            # Segmentation too aggressive -- keep everything near
            warnings.warn(
                "[Step 3] Segmentation removed too many points. "
                "Reverting to depth-threshold mask.", stacklevel=2
            )
            P_filtered   = pts_near
            C_filtered   = col_near
            fg_mask_sub  = near_mask.copy()

    # Build full-resolution mask
    fg_mask = np.zeros(N, dtype=bool)
    fg_mask[idx[fg_mask_sub]] = True

    n_fg = int(fg_mask.sum())
    print(
        f"[Step 3] Segmentation complete -- {n_fg:,} foreground points "
        f"({100*n_fg/N:.1f}% of input)"
    )
    return (
        points[fg_mask].astype(np.float32),
        colors[fg_mask].astype(np.float32),
        fg_mask,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _find_background_cluster(
    pts:    np.ndarray,
    labels: np.ndarray,
    unique_labels: set,
) -> int:
    """
    Identify the background cluster by highest planarity score.

    Planarity = (lambda1 - lambda2) / (lambda1 + lambda2 + lambda3)   (PCA on 3-D cluster points)
    The flattest cluster with the most points is the bin floor/wall.
    """
    best_label    = -1
    best_score    = -np.inf

    for lbl in unique_labels:
        mask  = labels == lbl
        count = mask.sum()
        cluster_pts = pts[mask]

        if count < 10:
            continue

        # PCA on cluster XYZ
        pca = PCA(n_components=3)
        pca.fit(cluster_pts)
        ev = pca.explained_variance_

        # Planarity: small 3rd eigenvalue -> flat surface
        planarity = 1.0 - (ev[2] / (ev[0] + 1e-9))
        # Weight by size (big flat cluster = background)
        score = planarity * np.log1p(count)

        if score > best_score:
            best_score = score
            best_label = lbl

    return best_label
