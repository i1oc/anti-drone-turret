"""Design of the arming cards: thick black frame + inner symbol.

This module is the ONLY definition of the design. It is used by:
  * cards/make_cards.py, to generate the printable PDF;
  * src/card_detector.py, which knows the frame proportions;
  * the tests, which render synthetic cards.

All measurements are in FRAME units (outer side of the frame = 1).
The detector "straightens" the frame into a 200x200 px square, so in the
rectified image 0.14 corresponds to 28 px.

Design criteria (they count towards the grade):
  * thick black frame: a quadrilateral with a white hole inside, which is rare
    in the scene, and its 4 corners give the homography to correct perspective;
  * white outer margin: to hold the card without covering the frame and so
    that the frame contrasts with any background;
  * rotation-invariant symbols: they are told apart by number of pieces, number
    of holes, fill relative to the enclosing circle, convexity and number of
    vertices, so in-plane rotation of the card does not matter;
  * every symbol differs from the others in at least one feature by a wide margin.
"""
from __future__ import annotations

import cv2
import numpy as np

FRAME_THICKNESS = 0.14   # thickness of the black frame (fraction of the frame side)
OUTER_MARGIN = 0.10      # outer white margin (fraction of the frame side)
SYMBOL_RADIUS = 0.27     # maximum symbol radius; the white hole has half-side 0.36

# Ratio hole area / outer frame area (detector filter)
INNER_AREA_RATIO = (1 - 2 * FRAME_THICKNESS) ** 2  # = 0.5184

# Class names (order does not matter; the arming code lives in config.yaml)
CIRCLE = "CIRCLE"
BULLSEYE = "BULLSEYE"
TRIANGLE = "TRIANGLE"
DIAMOND = "DIAMOND"
PENTAGON = "PENTAGON"
BARS = "BARS"
DOTS = "DOTS"
CANCEL = "CANCEL"
UNKNOWN = "UNKNOWN"
SYMBOLS = [CIRCLE, BULLSEYE, TRIANGLE, DIAMOND, PENTAGON, BARS, DOTS, CANCEL]


def _regular_polygon(n: int, radius: float, rot_deg: float) -> np.ndarray:
    """Regular polygon centred at (0.5, 0.5), in frame units."""
    a = np.deg2rad(rot_deg) + 2 * np.pi * np.arange(n) / n
    return np.column_stack([0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)])


def _shapes(symbol: str) -> list[tuple]:
    """Symbol primitives: ("circle", cx, cy, r, color) or ("poly", pts, color).

    color 0 = black (ink), 255 = white (to cut holes).
    """
    r = SYMBOL_RADIUS
    if symbol == CIRCLE:
        return [("circle", 0.5, 0.5, 0.25, 0)]
    if symbol == BULLSEYE:  # ring with a dot in the centre: 2 pieces, 1 hole
        # 0.11 ring-dot gap so they do not merge when blurred
        return [("circle", 0.5, 0.5, r, 0), ("circle", 0.5, 0.5, 0.19, 255), ("circle", 0.5, 0.5, 0.08, 0)]
    if symbol == TRIANGLE:
        return [("poly", _regular_polygon(3, 0.29, -90), 0)]
    if symbol == DIAMOND:  # square rotated 45°
        return [("poly", _regular_polygon(4, r, -90), 0)]
    if symbol == PENTAGON:
        return [("poly", _regular_polygon(5, 0.28, -90), 0)]
    if symbol == BARS:  # 3 vertical bars: 3 elongated pieces
        w, h = 0.07, 0.48   # 0.10 gap between bars so they do not merge when blurred
        out = []
        for cx in (0.5 - 0.17, 0.5, 0.5 + 0.17):
            out.append(("poly", np.array([[cx - w / 2, 0.5 - h / 2], [cx + w / 2, 0.5 - h / 2],
                                          [cx + w / 2, 0.5 + h / 2], [cx - w / 2, 0.5 + h / 2]]), 0))
        return out
    if symbol == DOTS:  # 4 dots in a square: 4 round pieces
        d = 0.14
        return [("circle", 0.5 + sx * d, 0.5 + sy * d, 0.075, 0) for sx in (-1, 1) for sy in (-1, 1)]
    if symbol == CANCEL:  # X cross: one very concave piece
        L, t = 0.30, 0.055   # half-length and half-thickness of each arm
        out = []
        for ang in (45, -45):
            c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            pts = np.array([[-L, -t], [L, -t], [L, t], [-L, t]])
            pts = pts @ np.array([[c, s], [-s, c]]) + 0.5
            out.append(("poly", pts, 0))
        return out
    raise ValueError(f"Unknown symbol: {symbol}")


def render_card(symbol: str | None, size_px: int = 400, margin: bool = True) -> np.ndarray:
    """Grayscale (uint8) image of the card. symbol=None draws only the frame.

    margin=True includes the outer white margin (full card, as printed);
    margin=False returns only the frame (what the detector sees after
    rectification).
    """
    m = OUTER_MARGIN if margin else 0.0
    total = 1 + 2 * m
    ss = 4  # supersampling for smooth edges (antialiasing)
    S = size_px * ss
    scale = S / total
    img = np.full((S, S), 255, np.uint8)

    def to_px(p):
        return (np.asarray(p, float) + m) * scale

    a, b = to_px([0, 0]), to_px([1, 1])
    cv2.rectangle(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 0, -1)
    a, b = to_px([FRAME_THICKNESS] * 2), to_px([1 - FRAME_THICKNESS] * 2)
    cv2.rectangle(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 255, -1)

    if symbol is not None:
        for sh in _shapes(symbol):
            if sh[0] == "circle":
                _, cx, cy, r, col = sh
                c = to_px([cx, cy])
                cv2.circle(img, (int(round(c[0])), int(round(c[1]))), int(round(r * scale)), col, -1, cv2.LINE_AA)
            else:
                _, pts, col = sh
                cv2.fillPoly(img, [np.round(to_px(pts)).astype(np.int32)], col, cv2.LINE_AA)
    return cv2.resize(img, (size_px, size_px), interpolation=cv2.INTER_AREA)
