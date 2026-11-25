"""Ball detector (classical colour segmentation), one class per ball colour.

Pipeline per frame:
    BGR -> Gaussian blur -> HSV (hue separates the colour from the lighting)
    -> for each colour: cv2.inRange with one or more [lower, upper] ranges
    -> morphology: opening removes specks, closing fills the specular highlight
       (and, with a larger kernel, the black seams of a basketball)
    -> external contours -> filters: area >= min_area,
       circularity 4*pi*A/P^2 >= min_circularity (circle = 1, square 0.79) and
       fill = A / area of the enclosing circle >= min_fill (circle ~0.9, square 0.64,
       a ball stretched by motion blur to an ellipse of axes ratio a gives ~a)
    -> [optional, per colour] motion confirmation
    -> minEnclosingCircle -> Detection(x, y, r, colour)

The balls (fluorescent tennis ball, orange basketball) are separated by hue.
Dark lines on a ball (the seams of a basketball) that reach its edge leave
notches open to the outside that no closing fills. A ball is convex, so with
`use_hull: true` the shape is measured on the convex hull of the blob: the
notches vanish, while a rectangle stays a rectangle.
When the ball is close to the camera a seam can be wide enough to split the
ball into separate blobs (a basketball cut in two halves). With `group_px > 0`
the fragments of the same colour closer than group_px are grouped (dilation +
connected regions) and the shape is measured on the convex hull of the whole
group, so the halves form one round ball again.

Per-colour option `motion_confirm: true` keeps a blob only if enough of it is
moving according to Zivkovic's MOG2 background subtractor (computed at half
resolution). It is meant for colours that the background also has (e.g. a
white ball against white walls); it was tested with a white/black football,
which was finally dropped. MOG2 only runs if some colour asks for it.

Each colour in config.yaml is either a list of ranges or a dict:
    {ranges: [...], close_px: 7, use_hull: false, group_px: 0, motion_confirm: false,
     min_circularity: .., min_fill: ..}
Ranges can be tuned with tools/pick_colors.py (click on the ball), which writes
config/colors.yaml; those ranges override the ones in config.yaml.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from tracking import Detection


@dataclass
class ColorSpec:
    ranges: list                 # [(lower hsv, upper hsv), ...] as uint8 arrays
    close_px: int = 7            # closing kernel (px); larger fills bigger dark patches
    motion_confirm: bool = False
    min_circularity: float | None = None   # None = detector default
    min_fill: float | None = None
    use_hull: bool = False       # measure the shape on the convex hull (ball with dark patches)
    group_px: int = 0            # group fragments closer than this before taking the hull (0 = off)


def parse_color(spec) -> ColorSpec:
    """Colour entry of the config (list of ranges or dict) -> ColorSpec."""
    if isinstance(spec, ColorSpec):
        return spec
    d = {"ranges": spec} if isinstance(spec, list) else dict(spec)
    ranges = [(np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in d["ranges"]]
    return ColorSpec(ranges, int(d.get("close_px", 7)), bool(d.get("motion_confirm", False)),
                     d.get("min_circularity"), d.get("min_fill"), bool(d.get("use_hull", False)),
                     int(d.get("group_px", 0)))


class BallDetector:
    def __init__(self, colors: dict, min_area: float = 80.0, min_circularity: float = 0.82,
                 blur: int = 5, motion_min_frac: float = 0.15, min_fill: float = 0.65):
        self.colors: dict[str, ColorSpec] = {name: parse_color(s) for name, s in colors.items()}
        self.min_area = min_area
        self.min_circularity = min_circularity
        self.min_fill = min_fill
        self.blur = blur | 1
        self.motion_min_frac = motion_min_frac
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernels: dict[int, np.ndarray] = {}
        self.mog2 = None
        if any(s.motion_confirm for s in self.colors.values()):
            self.mog2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
        self.timings: dict[str, float] = {}

    @classmethod
    def from_config(cls, cfg: dict, root: Path | None = None) -> "BallDetector":
        t = cfg.get("tracker", {})
        colors = {name: parse_color(s) for name, s in t.get("colors", {}).items()}
        colors_file = t.get("colors_file")
        if colors_file and root is not None and (root / colors_file).exists():
            with open(root / colors_file, encoding="utf-8") as f:
                tuned = yaml.safe_load(f) or {}
            for name, ranges in tuned.items():   # tuned ranges override the defaults, options are kept
                if name in colors:
                    colors[name].ranges = parse_color(ranges).ranges
        return cls(colors, t.get("min_area", 80.0), t.get("min_circularity", 0.82), t.get("blur", 5),
                   t.get("motion_min_frac", 0.15), t.get("min_fill", 0.65))

    def set_ranges(self, name: str, ranges: list) -> None:
        self.colors[name].ranges = parse_color(ranges).ranges

    def _close_kernel(self, px: int) -> np.ndarray:
        if px not in self._kernels:
            self._kernels[px] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px | 1, px | 1))
        return self._kernels[px]

    def color_mask(self, hsv: np.ndarray, name: str) -> np.ndarray:
        mask = None
        for lo, hi in self.colors[name].ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        return mask

    def _grouped_hulls(self, mask: np.ndarray, group_px: int) -> list[np.ndarray]:
        """Convex hull of each group of fragments closer than group_px (e.g. a ball split by a seam)."""
        grown = cv2.dilate(mask, self._close_kernel(group_px))
        groups, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hulls = []
        for g in groups:
            x, y, w, h = cv2.boundingRect(g)
            region = np.zeros((h, w), np.uint8)
            cv2.drawContours(region, [g], -1, 255, -1, offset=(-x, -y))
            pts = cv2.findNonZero(cv2.bitwise_and(mask[y:y + h, x:x + w], region))
            if pts is not None:
                hulls.append(cv2.convexHull(pts + np.array([[x, y]], np.int32)))
        return hulls

    def motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """MOG2 foreground at half resolution, resized back (≈4x cheaper than full size)."""
        small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
        fg = self.mog2.apply(small)
        return cv2.resize(fg, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    def detect(self, frame: np.ndarray, debug: dict | None = None) -> list[Detection]:
        t0 = time.perf_counter()
        blurred = cv2.GaussianBlur(frame, (self.blur, self.blur), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        motion = self.motion_mask(frame) if self.mog2 is not None else None
        out: list[Detection] = []
        if debug is not None:
            debug["1_hsv_hue"] = cv2.applyColorMap((hsv[..., 0].astype(np.uint16) * 255 // 180).astype(np.uint8),
                                                   cv2.COLORMAP_HSV)
            if motion is not None:
                debug["1_motion_mog2"] = motion
            vis = frame.copy()
        for name, spec in self.colors.items():
            min_circ = spec.min_circularity if spec.min_circularity is not None else self.min_circularity
            min_fill = spec.min_fill if spec.min_fill is not None else self.min_fill
            raw = self.color_mask(hsv, name)
            mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, self.k_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel(spec.close_px))
            if spec.group_px > 0:
                contours = self._grouped_hulls(mask, spec.group_px)
            else:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if spec.use_hull:
                    c = cv2.convexHull(c)
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue
                peri = cv2.arcLength(c, True)
                circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
                (x, y), r = cv2.minEnclosingCircle(c)
                fill = area / (np.pi * r * r) if r > 0 else 0.0
                ok = circ >= min_circ and fill >= min_fill
                moving = None
                if ok and spec.motion_confirm and motion is not None:
                    m = np.zeros(mask.shape, np.uint8)
                    cv2.drawContours(m, [c], -1, 255, -1)
                    inside = cv2.countNonZero(m)
                    moving = cv2.countNonZero(cv2.bitwise_and(motion, m)) / max(inside, 1)
                    ok = moving >= self.motion_min_frac
                if ok:
                    out.append(Detection(float(x), float(y), float(r), name))
                if debug is not None:
                    cv2.drawContours(vis, [c], -1, (0, 255, 0) if ok else (0, 0, 255), 2)
                    txt = f"{name} c={circ:.2f} f={fill:.2f}" + (f" mov={moving:.2f}" if moving is not None else "")
                    cv2.putText(vis, txt, (int(x - r), int(y - r) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1, cv2.LINE_AA)
            if debug is not None:
                debug[f"2_mask_{name}"] = raw
                debug[f"3_mask_clean_{name}"] = mask
        if debug is not None:
            debug["4_contours"] = vis
        self.timings = {"detect_ms": (time.perf_counter() - t0) * 1000}
        return out


def hsv_ranges_from_samples(hsv_pixels: np.ndarray, min_half_width: int = 6, max_half_width: int = 22) -> list:
    """HSV ranges that cover a set of sampled pixels (N, 3) of one colour.

    Chromatic colour: the hue is circular (0..180 in OpenCV), so its centre is a
    circular mean and a range crossing 0/180 is split in two. S and V only get a
    lower bound: brighter/more saturated is always fine.
    Achromatic colour (median saturation < 35, e.g. a white ball): any hue,
    saturation below a bound and value above a bound. The bound is low on
    purpose: an overexposed, pale part of a coloured ball still has S ~50.
    """
    px = np.asarray(hsv_pixels, float).reshape(-1, 3)
    if np.median(px[:, 1]) < 35:
        s_hi = int(np.clip(np.percentile(px[:, 1], 95) + 25, 30, 90))
        v_lo = int(np.clip(np.percentile(px[:, 2], 5) - 30, 100, 230))
        return [[[0, 0, v_lo], [180, s_hi, 255]]]
    ang = px[:, 0] * (2 * np.pi / 180.0)
    centre = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180
    dev = np.abs(((px[:, 0] - centre + 90) % 180) - 90)       # circular distance in hue units
    half = int(np.clip(np.percentile(dev, 95) + 4, min_half_width, max_half_width))
    s_lo = int(np.clip(np.percentile(px[:, 1], 5) - 20, 30, 200))   # pale lit side of a ball: S ~50
    v_lo = int(np.clip(np.percentile(px[:, 2], 5) - 40, 30, 200))
    lo, hi = int(round(centre)) - half, int(round(centre)) + half
    if lo < 0:
        return [[[0, s_lo, v_lo], [hi, 255, 255]], [[180 + lo, s_lo, v_lo], [180, 255, 255]]]
    if hi > 180:
        return [[[lo, s_lo, v_lo], [180, 255, 255]], [[0, s_lo, v_lo], [hi - 180, 255, 255]]]
    return [[[lo, s_lo, v_lo], [hi, 255, 255]]]
