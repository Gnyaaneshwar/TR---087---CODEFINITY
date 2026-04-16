"""
generate_test_data.py
======================
Generates synthetic RGB and depth test images for the grasp planning pipeline.

Produces:
  data/sample_rgb.png   — 480×640 RGB with coloured spheres on dark background
  data/sample_depth.png — matching uint16 depth image (Gaussian depth blobs)

Usage:
    python data/generate_test_data.py
"""

import os
import sys
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
H, W       = 480, 640
N_OBJECTS  = 6       # number of simulated objects in the bin
SEED       = 7


def generate_test_rgb_depth(seed: int = SEED):
    rng = np.random.default_rng(seed)

    # ── RGB image ──────────────────────────────────────────────────────────
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 30    # dark bin background

    # Random object colours
    obj_colours = [
        (220, 80,  80),   # red
        (80,  200, 100),  # green
        (80,  120, 220),  # blue
        (220, 200, 60),   # yellow
        (180, 80,  220),  # purple
        (60,  200, 200),  # cyan
    ]

    # Object positions (pixel centres) and radii
    centres = rng.integers(80, [W - 80, H - 80], size=(N_OBJECTS, 2))
    radii   = rng.integers(35, 75, size=N_OBJECTS)
    depths_m = rng.uniform(0.40, 0.90, size=N_OBJECTS)   # metres from camera

    for i, (cx, cy) in enumerate(centres):
        r     = int(radii[i])
        color = obj_colours[i % len(obj_colours)]
        # Draw filled circle (object silhouette)
        cv2.circle(rgb, (int(cx), int(cy)), r, color, -1)
        # Add subtle shading gradient
        for dr in range(r, 0, -5):
            alpha      = 1.0 - dr / r
            shade_col  = tuple(int(c * alpha + 200 * (1 - alpha)) for c in color)
            cv2.circle(rgb, (int(cx), int(cy)), dr, shade_col, 2)

    # Add some noise + blur for realism
    noise = rng.integers(-8, 8, rgb.shape, dtype=np.int16)
    rgb   = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    rgb   = cv2.GaussianBlur(rgb, (3, 3), 0.8)

    # ── Depth image ────────────────────────────────────────────────────────
    # Background depth = 1.2 m (bin floor)
    depth_m = np.full((H, W), 1.20, dtype=np.float32)

    yy, xx = np.mgrid[0:H, 0:W]

    for i, (cx, cy) in enumerate(centres):
        r   = float(radii[i])
        d   = float(depths_m[i])
        # Gaussian blob for each object
        dist_sq = ((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
        blob    = np.exp(-dist_sq / (2 * r ** 2))
        # Object is closer than background
        depth_m = np.where(blob > 0.3, d + (1 - blob) * 0.05, depth_m)

    # Add small noise
    depth_m += rng.uniform(-0.005, 0.005, (H, W)).astype(np.float32)
    depth_m  = np.clip(depth_m, 0.01, 2.0)

    # Introduce a 3% random missing region (NaN → 0 in uint16)
    missing_mask = rng.random((H, W)) < 0.03
    depth_m[missing_mask] = 0.0

    # Convert to uint16 millimetres for PNG storage
    depth_mm = (depth_m * 1000).astype(np.uint16)

    return rgb, depth_mm, depth_m


def save_images(rgb, depth_mm):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rgb_path   = os.path.join(OUTPUT_DIR, "sample_rgb.png")
    depth_path = os.path.join(OUTPUT_DIR, "sample_depth.png")

    cv2.imwrite(rgb_path,   cv2.cvtColor(rgb,      cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, depth_mm)

    print(f"[DataGen] RGB   -> {os.path.abspath(rgb_path)}")
    print(f"[DataGen] Depth -> {os.path.abspath(depth_path)}")
    print(f"[DataGen] Image size: {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"[DataGen] Depth range: {depth_mm.min()} - {depth_mm.max()} mm")
    return rgb_path, depth_path


if __name__ == "__main__":
    rgb, depth_mm, _ = generate_test_rgb_depth()
    save_images(rgb, depth_mm)
    print("[DataGen] Done — test data generated successfully.")
