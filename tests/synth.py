"""Cámara virtual para los tests: genera vistas del tablero con K y distorsión conocidas.

Permite comprobar la calibración sin cámara física: se conoce la verdad y se
mide cuánto se acerca lo estimado.
"""
from __future__ import annotations

import cv2
import numpy as np

K_TRUE = np.array([[620.0, 0.0, 318.0], [0.0, 615.0, 243.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.25, 0.08, 0.001, -0.0015, 0.0])
SIZE = (640, 480)
SQUARES = (11, 8)
SQUARE_MM = 20.0
PX_PER_SQUARE = 40


def random_pose(rng, K=K_TRUE, dist=DIST_TRUE, size=SIZE, squares=SQUARES, square=SQUARE_MM,
                max_angle_deg=35.0, z_range=(350.0, 650.0)):
    """Pose aleatoria (rvec, tvec) con el tablero entero (y un cuadro de margen) dentro de la imagen."""
    W, H = squares[0] * square, squares[1] * square
    # Contorno exterior del papel (un cuadro de margen), relativo a la primera esquina interior
    outline = np.array([[-2 * square, -2 * square, 0], [W, -2 * square, 0],
                        [W, H, 0], [-2 * square, H, 0]], np.float64)
    center = np.array([W / 2 - square, H / 2 - square, 0.0])
    for _ in range(1000):
        ang = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg, 3) * np.array([1, 1, 0.5]))
        R, _ = cv2.Rodrigues(ang)
        z = rng.uniform(*z_range)
        x = rng.uniform(-0.35, 0.35) * z
        y = rng.uniform(-0.25, 0.25) * z
        tvec = np.array([x, y, z]) - R @ center
        rvec = cv2.Rodrigues(R)[0]
        proj, _ = cv2.projectPoints(outline, rvec, tvec, K, dist)
        p = proj.reshape(-1, 2)
        if (p[:, 0] > 5).all() and (p[:, 0] < size[0] - 5).all() and \
           (p[:, 1] > 5).all() and (p[:, 1] < size[1] - 5).all():
            return rvec.ravel(), tvec.ravel()
    raise RuntimeError("No se encontró una pose válida")


def render_board(squares=SQUARES, pps=PX_PER_SQUARE):
    """Imagen del tablero (blanco/negro) con un cuadro de margen blanco alrededor."""
    sx, sy = squares
    img = np.full(((sy + 2) * pps, (sx + 2) * pps), 255, np.uint8)
    for r in range(sy):
        for c in range(sx):
            if (r + c) % 2 == 0:
                img[(r + 1) * pps:(r + 2) * pps, (c + 1) * pps:(c + 2) * pps] = 0
    return img


def render_view(rvec, tvec, K=K_TRUE, dist=DIST_TRUE, size=SIZE, squares=SQUARES,
                square=SQUARE_MM, pps=PX_PER_SQUARE, noise=2.0, rng=None):
    """Imagen en gris que vería la cámara (K, dist) con el tablero en la pose dada.

    Para cada píxel de salida: se quita la distorsión (undistortPoints), se corta
    el rayo con el plano Z = 0 del tablero y se muestrea la textura del tablero.
    """
    board = render_board(squares, pps)
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pix = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    ray = cv2.undistortPoints(pix, K, dist).reshape(-1, 2)
    d = np.column_stack([ray, np.ones(len(ray))])           # dirección del rayo (cámara)
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    t = np.asarray(tvec, np.float64).ravel()
    n = R[:, 2]                                             # normal del tablero en la cámara
    s = (n @ t) / (d @ n)
    P = d * s[:, None]                                      # punto 3D en la cámara
    Xb = (P - t) @ R                                        # = R^T (P - t): coordenadas del tablero (mm)
    # mm -> píxel de la textura. La esquina interior (0, 0) está en el píxel 2*pps - 0.5
    map_x = (Xb[:, 0] / square * pps + 2 * pps - 0.5).reshape(h, w).astype(np.float32)
    map_y = (Xb[:, 1] / square * pps + 2 * pps - 0.5).reshape(h, w).astype(np.float32)
    map_x[(s <= 0).reshape(h, w)] = -1
    img = cv2.remap(board, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=150)
    img = cv2.GaussianBlur(img, (3, 3), 0.7)                # desenfoque leve de la óptica
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img
