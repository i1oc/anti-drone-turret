"""On-screen panels (HUD). Drawing only: the logic lives in decoder.py, tracker, etc."""
from __future__ import annotations

import cv2
import numpy as np

from decoder import ARMED, ENTERING, LOCKED, PENALTY, SequenceDecoder

STATE_COLORS = {  # BGR
    LOCKED: (40, 40, 230),
    ENTERING: (0, 190, 255),
    ARMED: (60, 200, 60),
    PENALTY: (200, 60, 200),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, txt, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, txt, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_decoder_panel(img: np.ndarray, dec: SequenceDecoder, now: float) -> None:
    """Security panel (top right): state, stored cards, reading progress, message and timers.

    The stored cards are shown as filled slots, not as symbols, so the code cannot
    be read off the screen by someone watching.
    """
    h, w = img.shape[:2]
    pw, ph = 230, 118
    x0, y0 = w - pw - 8, 8
    roi = img[y0:y0 + ph, x0:x0 + pw]
    roi[:] = (roi * 0.35).astype(np.uint8)                      # translucent background
    color = STATE_COLORS.get(dec.state, (255, 255, 255))
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), color, 2)
    _text(img, dec.state, (x0 + 10, y0 + 26), 0.75, color, 2)

    rem = dec.remaining(now)
    if dec.state in (PENALTY, ENTERING) and rem > 0:
        _text(img, f"{rem:4.1f} s", (x0 + pw - 70, y0 + 26), 0.55)

    # Slots: one per symbol of the code, filled when a card is stored
    n = len(dec.code)
    for k in range(n):
        sx = x0 + 10 + k * 36
        filled = k < len(dec.entered) or dec.state == ARMED
        cv2.rectangle(img, (sx, y0 + 38), (sx + 28, y0 + 60), color if filled else (180, 180, 180),
                      -1 if filled else 1)

    # Progress of the card being held (reading)
    label, prog = dec.hold_progress()
    by = y0 + 72
    cv2.rectangle(img, (x0 + 10, by), (x0 + pw - 10, by + 8), (120, 120, 120), 1)
    if prog > 0:
        cv2.rectangle(img, (x0 + 10, by), (x0 + 10 + int((pw - 20) * prog), by + 8), (255, 255, 255), -1)
        _text(img, label, (x0 + 10, by + 24), 0.45)
    elif dec.awaiting_release and dec.state in (LOCKED, ENTERING):
        _text(img, "remove the card", (x0 + 10, by + 24), 0.45, (200, 200, 200))
    _text(img, dec.message[:32], (x0 + 10, y0 + ph - 8), 0.42, (230, 230, 230))


TRACK_COLORS = {"tennis": (0, 255, 200), "basketball": (0, 140, 255), "orange": (0, 140, 255),
                "red": (40, 40, 230), "green": (60, 200, 60), "blue": (230, 120, 30), "yellow": (0, 220, 240)}


def draw_tracks(img: np.ndarray, tracks, friend_colors=(), cam_model=None, undistorted: bool = False,
                ball_diameter_m=None, trail: bool = True) -> None:
    """Bounding box, id, colour, speed, trail and (if calibrated) angles and depth of each track.

    ball_diameter_m: one value for every ball or a dict colour -> diameter (m).
    Confirmed tracks: solid box. Lost (coasting on the Kalman prediction): thin box at the
    predicted position. Friendly colours (IFF) are labelled FRIEND.
    """
    for tr in tracks:
        col = TRACK_COLORS.get(tr.color, (255, 255, 255))
        lost = tr.state == "LOST"
        x0, y0, x1, y1 = tr.bbox()
        if trail and len(tr.history) > 1:
            pts = np.array(tr.history, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1 if lost else 2)
        px, py = tr.position
        vx, vy = tr.velocity
        cv2.arrowedLine(img, (int(px), int(py)), (int(px + 0.25 * vx), int(py + 0.25 * vy)), col, 2,
                        cv2.LINE_AA, tipLength=0.25)                     # where it will be in 0.25 s
        tag = f"#{tr.id} {tr.color}" + (" FRIEND" if tr.color in friend_colors else "") + (" (lost)" if lost else "")
        _text(img, tag, (x0, y0 - 6), 0.45, col)
        info = f"{tr.speed:.0f} px/s"
        if cam_model is not None:
            az, el = cam_model.pixel_to_angles([[px, py]], undistorted=undistorted)
            info += f"  az {np.degrees(az[0]):+.1f} el {np.degrees(el[0]):+.1f}"
            diam = ball_diameter_m.get(tr.color) if isinstance(ball_diameter_m, dict) else ball_diameter_m
            if diam:
                info += f"  Z {cam_model.depth_from_diameter(2 * tr.radius, diam):.2f} m"
        _text(img, info, (x0, y1 + 14), 0.4)
