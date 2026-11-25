"""FPS measurement with an exponential moving average and a performance overlay."""
from __future__ import annotations
import time

import cv2


class FPSCounter:
    """FPS smoothed with an exponential moving average (EMA).

    An EMA keeps the number from jumping between frames and is O(1) per frame.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.fps = 0.0
        self._last = None

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps


def draw_stats(frame, lines: list[str], corner: str = "bottom-left", scale: float = 0.42,
               margin: int = 8) -> None:
    """Draw small text lines on a dark box so they are readable over any image.

    corner: "bottom-left" (default) or "top-left". The lines keep their order
    (first line on top) in both cases.
    """
    font, thick, pad, gap = cv2.FONT_HERSHEY_SIMPLEX, 1, 4, 5
    sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
    if not sizes:
        return
    line_h = max(h for _, h in sizes)
    box_w = max(w for w, _ in sizes) + 2 * pad
    box_h = len(lines) * line_h + (len(lines) - 1) * gap + 2 * pad
    x0 = margin
    y0 = frame.shape[0] - margin - box_h if corner == "bottom-left" else margin
    roi = frame[y0:y0 + box_h, x0:x0 + box_w]
    roi[:] = (roi * 0.35).astype(roi.dtype)  # translucent background
    y = y0 + pad + line_h
    for text in lines:
        cv2.putText(frame, text, (x0 + pad, y), font, scale, (0, 255, 0), thick, cv2.LINE_AA)
        y += line_h + gap
