"""
step01_input_handler.py
========================
STEP 1 -- INPUT HANDLING

Responsibilities:
  1. Load or accept RGB (HxWx3) and depth (HxW) images.
  2. Validate identical resolution.
  3. Fill missing / NaN depth values via nearest-valid interpolation.
  4. Normalise RGB to [0, 1] (float32).
  5. Normalise depth via min-max scaling to [0, 1] (float32).

Returns
-------
rgb_norm   : np.ndarray (H, W, 3)  float32  -- RGB in [0, 1]
depth_norm : np.ndarray (H, W)     float32  -- depth in [0, 1]
depth_raw  : np.ndarray (H, W)     float32  -- depth in original units
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Union
import warnings


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_and_validate(
    rgb_input: Union[str, np.ndarray],
    depth_input: Union[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Step-1 pipeline: load -> validate -> repair -> normalise.

    Parameters
    ----------
    rgb_input   : file path (str) OR pre-loaded np.ndarray (H, W, 3) uint8/float
    depth_input : file path (str) OR pre-loaded np.ndarray (H, W) any numeric

    Returns
    -------
    rgb_norm   : float32 (H, W, 3) in [0, 1]
    depth_norm : float32 (H, W)    in [0, 1]
    depth_raw  : float32 (H, W)    in original units (after NaN repair)
    """
    # --- 1. Load ----------------------------------------------------------- #
    rgb   = _load_rgb(rgb_input)
    depth = _load_depth(depth_input)

    # --- 2. Validate resolution -------------------------------------------- #
    _validate_resolution(rgb, depth)

    # --- 3. Repair missing depth values ------------------------------------ #
    depth_raw = _fill_missing_depth(depth.astype(np.float32))

    # --- 4. Normalise ------------------------------------------------------- #
    rgb_norm   = _normalise_rgb(rgb)
    depth_norm = _normalise_depth(depth_raw)

    print(
        f"[Step 1] Input validated -- shape {rgb.shape[:2]}, "
        f"depth range [{depth_raw.min():.4f}, {depth_raw.max():.4f}]"
    )
    return rgb_norm, depth_norm, depth_raw


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _load_rgb(src: Union[str, np.ndarray]) -> np.ndarray:
    """Load RGB image from file path or validate existing array."""
    if isinstance(src, str):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"[Step 1] Cannot load RGB image: {src}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.asarray(src)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"[Step 1] RGB must be (H, W, 3), got shape {img.shape}"
        )
    return img


def _load_depth(src: Union[str, np.ndarray]) -> np.ndarray:
    """Load depth image from file path or validate existing array."""
    if isinstance(src, str):
        img = cv2.imread(src, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[Step 1] Cannot load depth image: {src}")
    else:
        img = np.asarray(src)

    if img.ndim != 2:
        raise ValueError(
            f"[Step 1] Depth must be (H, W), got shape {img.shape}"
        )
    return img.astype(np.float32)


def _validate_resolution(rgb: np.ndarray, depth: np.ndarray) -> None:
    """Assert RGB and depth share the same HxW."""
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"[Step 1] Resolution mismatch: "
            f"RGB {rgb.shape[:2]} != depth {depth.shape[:2]}"
        )


def _fill_missing_depth(depth: np.ndarray) -> np.ndarray:
    """
    Replace NaN, Inf, and zero values with the nearest valid depth.

    Uses scipy's Euclidean distance transform on a binary validity mask to
    propagate valid depth values into missing regions -- O(HxW), very fast.
    """
    # Treat 0, NaN, Inf as missing
    invalid_mask = ~np.isfinite(depth) | (depth == 0)

    if not invalid_mask.any():
        return depth                  # nothing to fix

    n_missing = int(invalid_mask.sum())
    total     = depth.size
    warnings.warn(
        f"[Step 1] Filling {n_missing}/{total} "
        f"({100*n_missing/total:.1f}%) missing depth pixels.",
        stacklevel=2,
    )

    # Distance transform: for each invalid pixel, find the nearest valid pixel
    _, nearest_idx = distance_transform_edt(
        invalid_mask, return_indices=True
    )
    repaired = depth[nearest_idx[0], nearest_idx[1]]
    return repaired.astype(np.float32)


def _normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    """Scale uint8 [0,255] or float [any] RGB to [0, 1] float32."""
    arr = rgb.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _normalise_depth(depth: np.ndarray) -> np.ndarray:
    """Min-max normalise depth to [0, 1] float32."""
    d_min = depth.min()
    d_max = depth.max()
    if d_max == d_min:
        warnings.warn("[Step 1] Depth is flat (all values identical). "
                      "Returning zero array.", stacklevel=2)
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)
