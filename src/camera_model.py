"""Calibrated camera model for the real-time system.

Loads calibration/calib.yaml and provides:
  * undistort(frame): distortion correction with precomputed maps
    (initUndistortRectifyMap once + remap per frame, much faster than
    cv2.undistort, which recomputes the maps every time);
  * pixel_to_angles: pixel -> (azimuth, elevation) of the ray, to aim the turret;
  * depth_from_diameter: approximate depth of a ball of known diameter.

Angle convention (pinhole camera, Z axis forward, Y down):
    x = (u - cx) / fx,  y = (v - cy) / fy           (normalized coordinates)
    az = atan(x)                                    (+ to the right)
    el = -atan(y / sqrt(1 + x^2))                   (+ upwards)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from calib_utils import load_calibration


def angles_from_normalized(x, y):
    """(azimuth, elevation) in radians of a ray with normalized coordinates (x, y, 1)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    az = np.arctan(x)
    el = -np.arctan(y / np.sqrt(1.0 + x * x))
    return az, el


def normalized_from_angles(az, el):
    """Inverse of angles_from_normalized (useful for tests and synthetic targets)."""
    az = np.asarray(az, dtype=np.float64)
    el = np.asarray(el, dtype=np.float64)
    x = np.tan(az)
    y = -np.tan(el) * np.sqrt(1.0 + x * x)
    return x, y


class CameraModel:
    def __init__(self, K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int],
                 alpha: float = 0.0):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.size = (int(image_size[0]), int(image_size[1]))  # (width, height)
        # Matrix of the undistorted image: alpha=0 crops the black borders, alpha=1 keeps them
        self.new_K, self.roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, self.size, alpha, self.size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.new_K, self.size, cv2.CV_16SC2)

    @classmethod
    def from_yaml(cls, path: str | Path, alpha: float = 0.0) -> "CameraModel":
        c = load_calibration(path)
        return cls(c["K"], c["dist"], c["image_size"], alpha)

    @property
    def focal(self) -> float:
        return 0.5 * (self.K[0, 0] + self.K[1, 1])

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"The frame is {w}x{h} but the calibration is {self.size[0]}x{self.size[1]}")
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def normalize(self, points_px, undistorted: bool = False) -> np.ndarray:
        """Pixels (N, 2) -> undistorted normalized coordinates (N, 2).

        undistorted=False: the points come from the original image (distortion is corrected).
        undistorted=True: they come from the image already corrected with undistort() (new_K is enough).
        """
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        if undistorted:
            K = self.new_K
            return np.column_stack(((pts[:, 0] - K[0, 2]) / K[0, 0], (pts[:, 1] - K[1, 2]) / K[1, 1]))
        return cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)

    def undistort_points(self, points_px) -> np.ndarray:
        """Pixels of the original image -> pixels in the undistorted image."""
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.new_K).reshape(-1, 2)

    def pixel_to_angles(self, points_px, undistorted: bool = False):
        """Pixels (N, 2) -> (az, el) in radians."""
        n = self.normalize(points_px, undistorted)
        return angles_from_normalized(n[:, 0], n[:, 1])

    def depth_from_diameter(self, diameter_px, diameter_m: float):
        """Depth Z = f * D / d (pinhole model, spherical object of diameter D)."""
        d = np.asarray(diameter_px, dtype=np.float64)
        return self.focal * diameter_m / d
