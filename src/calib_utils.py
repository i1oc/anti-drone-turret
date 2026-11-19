"""Chessboard calibration utilities (Zhang's method).

Used by the scripts in calibration/ (capture and offline calibration) and by the tests.
Based on lab 2, with two changes:
  * the 3D board points follow the same order returned by
    findChessboardCorners (row by row, X along the row), so the pose
    (rvec, tvec) has correctly oriented X/Y axes;
  * the cornerSubPix window adapts to the square size in the image so that
    neighbouring corners are not mixed when the board is far away.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import cv2
import numpy as np
import yaml

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def pattern_size(squares_x: int, squares_y: int) -> tuple[int, int]:
    """Number of inner corners (columns, rows) from the number of squares."""
    return squares_x - 1, squares_y - 1


def board_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    """3D coordinates (N, 3) of the inner corners in the board frame (Z = 0).

    The order matches findChessboardCorners: X (columns) first, then Y (rows).
    """
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return pts


def _subpix_window(corners: np.ndarray, pattern: tuple[int, int]) -> int:
    """Half-width of the cornerSubPix window: less than half a square in the image."""
    cols, _ = pattern
    grid = corners.reshape(-1, cols, 2)
    d = np.linalg.norm(np.diff(grid, axis=1), axis=2).min()  # smallest visible square, in px
    return int(np.clip(0.4 * d, 2, 7))


def find_corners(gray: np.ndarray, pattern: tuple[int, int], fast: bool = False,
                 subpix: bool = True) -> np.ndarray | None:
    """Detect the inner corners of the board. Returns (N, 1, 2) float32 or None.

    fast=True adds CALIB_CB_FAST_CHECK: quickly rejects frames without a board
    (for the live view). Calibration uses fast=False and subpixel refinement.
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not ok:
        return None
    if subpix:
        w = _subpix_window(corners, pattern)
        corners = cv2.cornerSubPix(gray, corners, (w, w), (-1, -1), SUBPIX_CRITERIA)
    return corners.astype(np.float32)


def calibrate(obj_points: list, img_points: list, image_size: tuple[int, int],
              flags: int = 0) -> dict:
    """Calibrate with cv2.calibrateCameraExtended.

    Returns a dict with rms, K, dist, rvecs, tvecs, standard deviations of the
    intrinsics (fx, fy, cx, cy, k1, k2, p1, p2, k3...) and per-image RMS error.
    """
    rms, K, dist, rvecs, tvecs, std_int, _, per_view = cv2.calibrateCameraExtended(
        obj_points, img_points, image_size, None, None, flags=flags)
    return {
        "rms": float(rms),
        "K": K,
        "dist": dist.ravel(),
        "rvecs": rvecs,
        "tvecs": tvecs,
        "std_intrinsics": std_int.ravel(),
        "per_view_rms": per_view.ravel(),
    }


def reprojection_residuals(obj: np.ndarray, img: np.ndarray, rvec, tvec,
                           K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Residuals (N, 2) = projection - detection, in pixels."""
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return (proj - img).reshape(-1, 2)


def validation_residuals(obj_points: list, img_points: list, K: np.ndarray,
                         dist: np.ndarray) -> list[np.ndarray]:
    """Residuals on images NOT used for calibration.

    With K and dist fixed, the pose of each image is estimated with solvePnP
    and the board is reprojected (as in question A.3 of lab 2).
    """
    out = []
    for obj, img in zip(obj_points, img_points):
        _, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        out.append(reprojection_residuals(obj, img, rvec, tvec, K, dist))
    return out


def rms_of(residuals: list[np.ndarray]) -> float:
    """Global RMS with the same definition as OpenCV: sqrt(sum ||e||^2 / number of points)."""
    r = np.concatenate([np.asarray(x).reshape(-1, 2) for x in residuals])
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


def split_train_val(n: int, val_fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible random split of n images into training and validation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    if n >= 2 and val_fraction > 0:
        n_val = min(max(n_val, 1), n - 1)
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def _to_builtin(v):
    """Convert numpy arrays and scalars (also nested) to Python types for YAML."""
    if isinstance(v, dict):
        return {k: _to_builtin(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_builtin(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def save_calibration(path: str | Path, K: np.ndarray, dist: np.ndarray,
                     image_size: tuple[int, int], **extra) -> None:
    """Save the calibration as readable YAML (lists of floats, not numpy objects)."""
    data = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "K": np.asarray(K, float).tolist(),
        "dist": np.asarray(dist, float).ravel().tolist(),
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    data.update(_to_builtin(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_calibration(path: str | Path) -> dict:
    """Load the calibration YAML. K (3x3) and dist (1D) are returned as numpy arrays."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["K"] = np.array(data["K"], dtype=np.float64)
    data["dist"] = np.array(data["dist"], dtype=np.float64)
    data["image_size"] = (data["image_width"], data["image_height"])
    return data
