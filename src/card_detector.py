"""Arming card detector (classical vision, no learning).

Stage 1 - Locate the frame (in the camera frame):
    gray -> Gaussian blur -> ADAPTIVE threshold (copes with uneven lighting)
    -> findContours(RETR_TREE) -> candidates = contours that
       * are approximated by 4 vertices (approxPolyDP, Douglas-Peucker) and are convex,
       * have a child contour (the white hole of the frame),
       * with a hole_area / frame_area ratio close to the design value (0.52).
    This last filter rejects almost any other quadrilateral in the scene.

Stage 2 - Rectify:
    homography from the 4 corners to a 200x200 square (warpPerspective).
    It checks that the rectified frame is dark and the hole is bright.

Stage 3 - Classify the symbol (on the rectified card):
    crop the hole -> OTSU threshold (the histogram is now bimodal: ink and
    paper) -> morphological cleanup -> contours with hierarchy (RETR_CCOMP) ->
    rotation-invariant features:
       pieces      number of black components
       holes       number of holes inside the pieces
       fill        area / area of the minimum enclosing circle (circle 1.00,
                   pentagon 0.76, diamond 0.64, triangle 0.41)
       solidity    area / convex hull area (X cross ~0.5, the rest ~1)
       vertices    number of "real" sides of approxPolyDP on the convex hull
                   (sides < 30 % of the longest are ignored: corners rounded by blur)
       elongation  long side / short side of the minimum rectangle of each piece
       tilt        angle of that rectangle relative to the frame, between 0° and 45°:
                   the diamond is at ~45° (vertices towards the frame sides); a block
                   of bars merged by blur, at ~0°
    -> rules -> symbol or UNKNOWN.
    Polygons are classified FIRST by number of vertices (a very stable feature)
    and fill only confirms with wide ranges: with blur the corners get rounded
    and fill goes up (the diamond goes from 0.67 to ~0.79).

The reference feature values come from the design geometry (card_symbols.py),
not from training: that is why the rules are explainable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from card_symbols import (BARS, BULLSEYE, CANCEL, CIRCLE, DIAMOND, DOTS, FRAME_THICKNESS,
                          INNER_AREA_RATIO, PENTAGON, TRIANGLE, UNKNOWN)

# Reference features (ideal geometry): fill circle 0.98 | pentagon 0.77 | diamond 0.67 |
# triangle 0.43. Blur raises them by up to ~0.08. Confirmation ranges by number of vertices:
FILL_RANGES = {3: (0.30, 0.66), 4: (0.52, 0.86), 5: (0.64, 0.92)}
POLYGONS = {3: TRIANGLE, 4: DIAMOND, 5: PENTAGON}
FILL_CIRCLE_MIN = 0.89      # circle: fill >= 0.92 always; the pentagon never exceeds 0.85
DIAMOND_TILT_MIN = 25.0     # the diamond is rotated ~45° relative to the frame (an aligned square, ~0°)
SIDE_MIN_FRAC = 0.30        # sides shorter than this (x longest side) do not count as a vertex
REGULARITY_MIN = 0.60       # shortest / longest side of a regular polygon (diamond >= 0.83, pentagon >= 0.67;
                            # a triangle with a rounded tip that looks 4-sided gives ~0.45)
SOLIDITY_CONVEX_MIN = 0.85  # convex shapes: solidity ~1; X cross ~0.5
SOLIDITY_CROSS_MAX = 0.70
BAR_ELONGATION_MIN = 3.0    # bars: ~6.9 ; dots: ~1
DOT_ELONGATION_MAX = 1.6
MIN_PIECE_FRAC = 0.004      # pieces smaller than this fraction of the hole are noise
MAX_SIDE_RATIO = 2.5        # longest / shortest side of the quadrilateral (a card tilted 60° gives 2)


@dataclass
class CardDetection:
    label: str
    quad: np.ndarray                 # (4, 2) frame corners in the image, clockwise
    features: dict = field(default_factory=dict)
    rectified: np.ndarray | None = None

    @property
    def center(self) -> tuple[float, float]:
        c = self.quad.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def area(self) -> float:
        return float(cv2.contourArea(self.quad.astype(np.float32)))


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points clockwise (in image coordinates, y pointing down) starting from the smallest x+y."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


class CardDetector:
    def __init__(self, roi_fraction: float = 0.8, min_area_frac: float = 0.004,
                 block_size: int = 31, threshold_c: float = 10, rectified_px: int = 200):
        self.roi_fraction = roi_fraction
        self.min_area_frac = min_area_frac
        self.block_size = block_size | 1  # must be odd
        self.threshold_c = threshold_c
        self.R = rectified_px
        self.timings: dict[str, float] = {}

    @classmethod
    def from_config(cls, cfg: dict) -> "CardDetector":
        c = cfg.get("cards", {})
        return cls(c.get("roi_fraction", 0.8), c.get("min_area_frac", 0.004), c.get("block_size", 31),
                   c.get("threshold_c", 10), c.get("rectified_px", 200))

    # ------------------------------------------------------------------ stage 1
    def roi(self, shape) -> tuple[int, int, int, int]:
        """Central rectangle (x0, y0, x1, y1) where cards are searched for."""
        h, w = shape[:2]
        mx, my = int(w * (1 - self.roi_fraction) / 2), int(h * (1 - self.roi_fraction) / 2)
        return mx, my, w - mx, h - my

    def find_quads(self, gray: np.ndarray, debug: dict | None = None) -> list[np.ndarray]:
        """Candidate frame quadrilaterals (full-image coordinates)."""
        x0, y0, x1, y1 = self.roi(gray.shape)
        sub = cv2.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 0)
        # THRESH_BINARY_INV: dark areas (frame, symbol) become white = foreground
        binary = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                       self.block_size, self.threshold_c)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_frac * binary.size
        quads = []
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                child = hierarchy[i][2]
                if area < min_area or child < 0:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                sides = np.linalg.norm(approx.reshape(4, 2) - np.roll(approx.reshape(4, 2), 1, axis=0), axis=1)
                if sides.max() > MAX_SIDE_RATIO * sides.min():  # too elongated to be a card
                    continue
                # Hole = the child with the largest area
                hole, j = 0.0, child
                while j >= 0:
                    hole = max(hole, cv2.contourArea(contours[j]))
                    j = hierarchy[j][0]  # next sibling
                ratio = hole / area
                if not (0.6 * INNER_AREA_RATIO < ratio < 1.35 * INNER_AREA_RATIO):
                    continue
                quads.append(order_corners(approx.reshape(4, 2) + [x0, y0]))
        if debug is not None:
            debug["1_gray"] = gray
            debug["2_adaptive_threshold"] = binary
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 1)
            cv2.drawContours(vis[y0:y1, x0:x1], contours, -1, (0, 140, 255), 1)
            for q in quads:
                cv2.polylines(vis, [q.astype(np.int32)], True, (0, 220, 0), 2)
            debug["3_contours"] = vis
        return quads

    # ------------------------------------------------------------------ stage 2
    def rectify(self, gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
        R = self.R
        dst = np.array([[0, 0], [R - 1, 0], [R - 1, R - 1], [0, R - 1]], np.float32)
        H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        return cv2.warpPerspective(gray, H, (R, R), flags=cv2.INTER_LINEAR)

    def frame_check(self, rect: np.ndarray) -> dict:
        """Check that the rectified image really is a black frame with a bright hole.

        Avoids mistaking the frame for the paper edge or the inner hole, which
        are also quadrilaterals: in those cases the "frame" band is not
        uniformly dark.
        """
        R, t = self.R, int(FRAME_THICKNESS * self.R)
        k = max(2, t // 4)  # skip the frame edges (blur, corner error)
        band = np.zeros((R, R), bool)
        band[k:t - k, k:R - k] = True          # top side
        band[R - t + k:R - k, k:R - k] = True  # bottom
        band[k:R - k, k:t - k] = True          # left
        band[k:R - k, R - t + k:R - k] = True  # right
        inner = rect[t + k:R - t - k, t + k:R - t - k]
        thr, _ = cv2.threshold(rect, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return {
            "frame_dark": float(np.mean(rect[band] < thr)),   # ~1 on a real frame
            "hole_bright": float(np.mean(inner > thr)),       # the circle, the largest symbol, leaves ~0.55
            "contrast": float(np.percentile(inner, 90) - np.median(rect[band])),
        }

    @staticmethod
    def is_frame(fc: dict) -> bool:
        return fc["frame_dark"] > 0.9 and fc["hole_bright"] > 0.35 and fc["contrast"] > 30

    # ------------------------------------------------------------------ stage 3
    def symbol_mask(self, rect: np.ndarray) -> np.ndarray:
        """Binary mask of the symbol (255 = ink) inside the frame hole."""
        R = self.R
        pad = int((FRAME_THICKNESS + 0.03) * R)       # leave out the frame and its edge
        inner = cv2.GaussianBlur(rect[pad:R - pad, pad:R - pad], (3, 3), 0)
        # If the hole is uniform (no symbol), Otsu would split noise: return empty
        if float(np.percentile(inner, 95) - np.percentile(inner, 5)) < 40:
            return np.zeros_like(inner)
        _, mask = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    @staticmethod
    def symbol_features(mask: np.ndarray) -> dict:
        """Rotation-invariant features of the symbol (see the module docstring)."""
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        min_area = MIN_PIECE_FRAC * mask.size
        H, W = mask.shape
        pieces, holes, border = [], 0, 0
        if hierarchy is not None:
            hier = hierarchy[0]
            for i, (c, h) in enumerate(zip(contours, hier)):
                if cv2.contourArea(c) < min_area:
                    continue
                if h[3] >= 0:
                    continue               # holes: counted with their piece
                x, y, w, hh = cv2.boundingRect(c)
                if x <= 1 or y <= 1 or x + w >= W - 1 or y + hh >= H - 1:
                    border += 1            # touches the hole edge: leftover frame or another object,
                    continue               # the design always leaves a margin around the symbol
                pieces.append(c)           # outer contour = piece
                j = h[2]                   # its children are its holes
                while j >= 0:
                    if cv2.contourArea(contours[j]) >= min_area:
                        holes += 1
                    j = hier[j][0]
        f = {"pieces": len(pieces), "holes": holes, "border": border}
        if not pieces:
            return f
        elong = []
        for c in pieces:
            (_, _), (w, h), _ = cv2.minAreaRect(c)
            elong.append(max(w, h) / max(min(w, h), 1e-6))
        f["elongation"] = [round(e, 2) for e in elong]
        big = max(pieces, key=cv2.contourArea)
        area = cv2.contourArea(big)
        _, r = cv2.minEnclosingCircle(big)
        hull = cv2.convexHull(big)
        f["fill"] = round(area / (np.pi * r * r), 3)
        f["solidity"] = round(area / max(cv2.contourArea(hull), 1e-6), 3)
        poly = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        sides = np.linalg.norm(poly - np.roll(poly, 1, axis=0), axis=1)
        (_, _), (_, _), ang = cv2.minAreaRect(big)
        f["tilt"] = round(float(45 - abs((ang % 90) - 45)), 1)   # 0° aligned with the frame, 45° diagonal
        real = sides[sides >= SIDE_MIN_FRAC * sides.max()]
        f["vertices"] = int(len(real))
        f["regularity"] = round(float(real.min() / real.max()), 3)  # equal sides -> 1
        if len(pieces) == 2:  # for the bullseye: is one piece inside the other?
            (c1, c2) = sorted(pieces, key=cv2.contourArea, reverse=True)
            m = cv2.moments(c2)
            p = (m["m10"] / max(m["m00"], 1e-6), m["m01"] / max(m["m00"], 1e-6))
            f["concentric"] = cv2.pointPolygonTest(c1, p, False) >= 0 or \
                np.linalg.norm(np.array(p) - cv2.minEnclosingCircle(c1)[0]) < 0.1 * mask.shape[0]
        return f

    @staticmethod
    def classify(f: dict) -> str:
        """Rules on the features. When in doubt, UNKNOWN."""
        n, holes = f.get("pieces", 0), f.get("holes", 0)
        if n == 2 and holes == 1 and f.get("concentric"):
            return BULLSEYE
        if n == 3 and holes == 0 and min(f["elongation"]) >= BAR_ELONGATION_MIN:
            return BARS
        if n == 4 and holes == 0 and max(f["elongation"]) <= DOT_ELONGATION_MAX:
            return DOTS
        if n != 1 or holes != 0:
            return UNKNOWN
        fill, sol, v = f["fill"], f["solidity"], f["vertices"]
        if sol <= SOLIDITY_CROSS_MAX:
            return CANCEL
        if sol < SOLIDITY_CONVEX_MIN:
            return UNKNOWN
        if fill >= FILL_CIRCLE_MIN:
            return CIRCLE if v >= 5 else UNKNOWN
        if v in POLYGONS:
            lo, hi = FILL_RANGES[v]
            ok = lo <= fill <= hi and f["regularity"] >= REGULARITY_MIN
            if v == 4:
                ok = ok and f["tilt"] >= DIAMOND_TILT_MIN
            return POLYGONS[v] if ok else UNKNOWN
        return UNKNOWN

    # ------------------------------------------------------------------ all together
    def detect(self, frame: np.ndarray, debug: dict | None = None) -> list[CardDetection]:
        """Detect and classify every card in the frame (BGR or grayscale).

        If a dict is passed as debug, it is filled with the intermediate images
        (for the "sequence of transformations" figure in the report).
        """
        t0 = time.perf_counter()
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quads = self.find_quads(gray, debug)
        t1 = time.perf_counter()
        out = []
        for q in quads:
            rect = self.rectify(gray, q)
            fc = self.frame_check(rect)
            if not self.is_frame(fc):  # paper edge, inner hole or some other quadrilateral
                continue
            mask = self.symbol_mask(rect)
            f = self.symbol_features(mask)
            if f["pieces"] == 0:  # empty frame (e.g. a picture on the wall): not a card
                continue
            f.update({k: round(v, 2) for k, v in fc.items()})
            det = CardDetection(self.classify(f), q, f, rect)
            out.append(det)
            if debug is not None and "4_rectified" not in debug:
                debug["4_rectified"] = rect
                debug["5_symbol"] = mask
        # Nested or duplicated cards (same frame detected twice): keep the largest
        out.sort(key=lambda d: d.area, reverse=True)
        kept: list[CardDetection] = []
        for d in out:
            if all(cv2.pointPolygonTest(k.quad.astype(np.float32), d.center, False) < 0 for k in kept):
                kept.append(d)
        self.timings = {"frame_ms": (t1 - t0) * 1000, "symbol_ms": (time.perf_counter() - t1) * 1000}
        return kept


def features_text(f: dict) -> str:
    """Short summary of the features for on-screen display."""
    t = f"p={f.get('pieces', 0)} h={f.get('holes', 0)}"
    if "fill" in f:
        t += (f" fill={f['fill']:.2f} sol={f['solidity']:.2f} v={f['vertices']}"
              f" reg={f['regularity']:.2f} tilt={f['tilt']:.0f}")
    return t


def draw_detections(img: np.ndarray, dets: list[CardDetection], roi=None, show_features: bool = True) -> None:
    """Draw the detected frame, its class and (optionally) its features on the image (in place)."""
    if roi is not None:
        cv2.rectangle(img, roi[:2], roi[2:], (120, 120, 120), 1)
    for d in dets:
        color = (0, 200, 0) if d.label != UNKNOWN else (0, 0, 255)
        cv2.polylines(img, [d.quad.astype(np.int32)], True, color, 2)
        x, y = d.quad[:, 0].min(), d.quad[:, 1].min()
        cv2.putText(img, d.label, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if show_features:
            yb = int(d.quad[:, 1].max()) + 16
            txt = features_text(d.features)
            cv2.putText(img, txt, (int(x), yb), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (int(x), yb), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
