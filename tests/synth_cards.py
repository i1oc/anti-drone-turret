"""Synthetic scenes with cards to test the detector without a camera.

The card (real design from card_symbols) is placed in 3D in front of a virtual
camera with random rotation and tilt, and projected onto a background with
objects, uneven lighting, blur and noise.
"""
from __future__ import annotations

import cv2
import numpy as np

from card_symbols import OUTER_MARGIN, render_card

K = np.array([[495.0, 0, 320.0], [0, 497.0, 240.0], [0, 0, 1]])  # similar to the real camera
SIZE = (640, 480)


def background(rng, size=SIZE) -> np.ndarray:
    """Grey background with a gradient, distractor rectangles/lines and texture."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = rng.uniform(60, 200) + rng.uniform(-40, 40) * (xx / w) + rng.uniform(-40, 40) * (yy / h)
    img = np.clip(base, 0, 255).astype(np.uint8)
    for _ in range(rng.integers(3, 8)):  # distractors: filled and hollow rectangles, lines
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        a, b = int(rng.integers(20, 120)), int(rng.integers(20, 120))
        col = int(rng.integers(0, 256))
        kind = rng.integers(3)
        if kind == 0:
            cv2.rectangle(img, (x, y), (x + a, y + b), col, -1)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + a, y + b), col, int(rng.integers(3, 15)))
        else:
            cv2.line(img, (x, y), (x + a, y - b), col, int(rng.integers(2, 10)))
    return img


def card_scene(symbol, rng, max_tilt_deg=40.0, side_px=(110, 230), size=SIZE,
               light=(0.55, 1.0), blur=(0.0, 1.2), noise=3.0):
    """Scene with one card. Returns (grayscale image, projected frame corners (4, 2))."""
    card = render_card(symbol, 600)                   # full card with margin
    S = card.shape[0]
    tot = 1 + 2 * OUTER_MARGIN
    w, h = size
    for _ in range(200):
        # Pose: free in-plane rotation, limited out-of-plane tilt
        rz = rng.uniform(-np.pi, np.pi)
        rx, ry = np.deg2rad(rng.uniform(-max_tilt_deg, max_tilt_deg, 2))
        R = cv2.Rodrigues(np.array([rx, ry, 0.0]))[0] @ cv2.Rodrigues(np.array([0, 0, rz]))[0]
        side = rng.uniform(*side_px)                  # card side (with margin) in px, seen head-on
        Z = K[0, 0] * 1.0 / side                      # card of side 1 at distance Z
        cx, cy = rng.uniform(0.35, 0.65) * w, rng.uniform(0.35, 0.65) * h
        t = np.array([(cx - K[0, 2]) * Z / K[0, 0], (cy - K[1, 2]) * Z / K[1, 1], Z])
        sq = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]])
        P = (sq @ R.T + t) @ K.T
        dst = P[:, :2] / P[:, 2:]
        if (dst[:, 0] < 5).any() or (dst[:, 0] > w - 5).any() or (dst[:, 1] < 5).any() or (dst[:, 1] > h - 5).any():
            continue
        src = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
        H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        img = background(rng, size)
        warped = cv2.warpPerspective(card, H, size, flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full_like(card, 255), H, size, flags=cv2.INTER_LINEAR)
        a = mask.astype(np.float32) / 255
        img = (img * (1 - a) + warped * a).astype(np.float32)
        # Uneven lighting (soft shadow), multiplicative
        yy, xx = np.mgrid[0:h, 0:w]
        ang = rng.uniform(0, 2 * np.pi)
        g = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
        g = (g - g.min()) / (g.max() - g.min() + 1e-9)
        img *= light[0] + (light[1] - light[0]) * g
        s = rng.uniform(*blur)
        if s > 0.1:
            img = cv2.GaussianBlur(img, (0, 0), s)
        img += rng.normal(0, noise, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # Frame corners (without the white margin)
        m = OUTER_MARGIN / tot
        fr = np.array([[m, m], [1 - m, m], [1 - m, 1 - m], [m, 1 - m]]) * S
        frame = cv2.perspectiveTransform(fr.reshape(-1, 1, 2).astype(np.float32), H).reshape(4, 2)
        return img, frame
    raise RuntimeError("No valid pose found")
