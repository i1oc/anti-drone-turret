"""Modelo de cámara calibrada para el sistema en tiempo real.

Carga calibration/calib.yaml y ofrece:
  * undistort(frame): corrección de distorsión con mapas precalculados
    (initUndistortRectifyMap una vez + remap por frame, mucho más rápido que
    cv2.undistort, que recalcula los mapas cada vez);
  * pixel_to_angles: píxel -> (azimut, elevación) del rayo, para apuntar la torreta;
  * depth_from_diameter: profundidad aproximada de una pelota de diámetro conocido.

Convención de ángulos (cámara pinhole, eje Z hacia delante, Y hacia abajo):
    x = (u - cx) / fx,  y = (v - cy) / fy           (coordenadas normalizadas)
    az = atan(x)                                    (+ a la derecha)
    el = -atan(y / sqrt(1 + x^2))                   (+ hacia arriba)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from calib_utils import load_calibration


def angles_from_normalized(x, y):
    """(azimut, elevación) en radianes de un rayo con coordenadas normalizadas (x, y, 1)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    az = np.arctan(x)
    el = -np.arctan(y / np.sqrt(1.0 + x * x))
    return az, el


def normalized_from_angles(az, el):
    """Inversa de angles_from_normalized (útil para tests y objetivos sintéticos)."""
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
        self.size = (int(image_size[0]), int(image_size[1]))  # (ancho, alto)
        # Matriz de la imagen corregida: alpha=0 recorta los bordes negros, alpha=1 los conserva
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
            raise ValueError(f"El frame es {w}x{h} pero la calibración es {self.size[0]}x{self.size[1]}")
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def normalize(self, points_px, undistorted: bool = False) -> np.ndarray:
        """Píxeles (N, 2) -> coordenadas normalizadas sin distorsión (N, 2).

        undistorted=False: los puntos vienen de la imagen original (se corrige la distorsión).
        undistorted=True: vienen de la imagen ya corregida con undistort() (basta con new_K).
        """
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        if undistorted:
            K = self.new_K
            return np.column_stack(((pts[:, 0] - K[0, 2]) / K[0, 0], (pts[:, 1] - K[1, 2]) / K[1, 1]))
        return cv2.undistortPoints(pts.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)

    def undistort_points(self, points_px) -> np.ndarray:
        """Píxeles de la imagen original -> píxeles en la imagen corregida."""
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.new_K).reshape(-1, 2)

    def pixel_to_angles(self, points_px, undistorted: bool = False):
        """Píxeles (N, 2) -> (az, el) en radianes."""
        n = self.normalize(points_px, undistorted)
        return angles_from_normalized(n[:, 0], n[:, 1])

    def depth_from_diameter(self, diameter_px, diameter_m: float):
        """Profundidad Z = f * D / d (modelo pinhole, objeto esférico de diámetro D)."""
        d = np.asarray(diameter_px, dtype=np.float64)
        return self.focal * diameter_m / d
