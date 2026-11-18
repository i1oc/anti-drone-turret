"""Utilidades de calibración con tablero de ajedrez (método de Zhang).

Las usan los scripts de calibration/ (captura y calibración offline) y los tests.
Basado en la práctica 2, con dos cambios:
  * los puntos 3D del tablero siguen el mismo orden que devuelve
    findChessboardCorners (fila a fila, X a lo largo de la fila), de modo que la
    pose (rvec, tvec) tiene los ejes X/Y bien orientados;
  * la ventana de cornerSubPix se adapta al tamaño del cuadro en la imagen para
    no mezclar esquinas vecinas cuando el tablero está lejos.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import cv2
import numpy as np
import yaml

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def pattern_size(squares_x: int, squares_y: int) -> tuple[int, int]:
    """Nº de esquinas interiores (columnas, filas) a partir del nº de cuadros."""
    return squares_x - 1, squares_y - 1


def board_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    """Coordenadas 3D (N, 3) de las esquinas interiores en el sistema del tablero (Z = 0).

    El orden coincide con el de findChessboardCorners: primero se recorre X
    (columnas) y luego Y (filas).
    """
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return pts


def _subpix_window(corners: np.ndarray, pattern: tuple[int, int]) -> int:
    """Semiancho de la ventana de cornerSubPix: menor que medio cuadro en la imagen."""
    cols, _ = pattern
    grid = corners.reshape(-1, cols, 2)
    d = np.linalg.norm(np.diff(grid, axis=1), axis=2).min()  # cuadro más pequeño visto, en px
    return int(np.clip(0.4 * d, 2, 7))


def find_corners(gray: np.ndarray, pattern: tuple[int, int], fast: bool = False,
                 subpix: bool = True) -> np.ndarray | None:
    """Detecta las esquinas interiores del tablero. Devuelve (N, 1, 2) float32 o None.

    fast=True añade CALIB_CB_FAST_CHECK: descarta enseguida los frames sin tablero
    (para la vista en vivo). Para calibrar se usa fast=False y refinado subpíxel.
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
    """Calibra con cv2.calibrateCameraExtended.

    Devuelve un dict con rms, K, dist, rvecs, tvecs, desviaciones típicas de los
    intrínsecos (fx, fy, cx, cy, k1, k2, p1, p2, k3...) y error RMS por imagen.
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
    """Residuos (N, 2) = proyección - detección, en píxeles."""
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return (proj - img).reshape(-1, 2)


def validation_residuals(obj_points: list, img_points: list, K: np.ndarray,
                         dist: np.ndarray) -> list[np.ndarray]:
    """Residuos en imágenes NO usadas para calibrar.

    Con K y dist fijos, la pose de cada imagen se estima con solvePnP y se
    reproyecta el tablero (como en la pregunta A.3 de la práctica 2).
    """
    out = []
    for obj, img in zip(obj_points, img_points):
        _, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
        out.append(reprojection_residuals(obj, img, rvec, tvec, K, dist))
    return out


def rms_of(residuals: list[np.ndarray]) -> float:
    """RMS global con la misma definición que OpenCV: sqrt(sum ||e||^2 / nº de puntos)."""
    r = np.concatenate([np.asarray(x).reshape(-1, 2) for x in residuals])
    return float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))


def split_train_val(n: int, val_fraction: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Partición aleatoria reproducible de n imágenes en entrenamiento y validación."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    if n >= 2 and val_fraction > 0:
        n_val = min(max(n_val, 1), n - 1)
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def _to_builtin(v):
    """Convierte arrays y escalares numpy (también anidados) a tipos de Python para YAML."""
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
    """Guarda la calibración en YAML legible (listas de floats, no objetos numpy)."""
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
    """Carga el YAML de calibración. K (3x3) y dist (1D) se devuelven como numpy."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["K"] = np.array(data["K"], dtype=np.float64)
    data["dist"] = np.array(data["dist"], dtype=np.float64)
    data["image_size"] = (data["image_width"], data["image_height"])
    return data
