"""
camera_intrinsics.py
Camera intrinsic parameters for RGB-D sensor backprojection.
Default values match a standard 640x480 RGB-D sensor (e.g., RealSense D415).
"""
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic parameters.

    Parameters
    ----------
    fx, fy : focal lengths in pixels
    cx, cy : principal point (image centre) in pixels
    scale  : depth scale factor -- converts raw depth units to metres
             e.g. 1000.0 if depth is stored in millimetres
    width, height : sensor resolution in pixels
    """
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5
    scale: float = 1.0          # assume depth already in metres after normalisation
    width: int = 640
    height: int = 480

    # ------------------------------------------------------------------ #
    # Convenience factory methods
    # ------------------------------------------------------------------ #
    @classmethod
    def realsense_d415_640x480(cls) -> "CameraIntrinsics":
        """Intel RealSense D415 at 640x480."""
        return cls(fx=617.0, fy=617.0, cx=320.0, cy=240.0,
                   scale=1000.0, width=640, height=480)

    @classmethod
    def kinect_v2_640x480(cls) -> "CameraIntrinsics":
        """Microsoft Kinect v2 at 640x480."""
        return cls(fx=525.0, fy=525.0, cx=319.5, cy=239.5,
                   scale=1000.0, width=640, height=480)

    @classmethod
    def from_dict(cls, d: dict) -> "CameraIntrinsics":
        return cls(**d)

    def to_dict(self) -> dict:
        return {
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "scale": self.scale,
            "width": self.width, "height": self.height,
        }
