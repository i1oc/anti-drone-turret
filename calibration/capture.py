"""Capture of chessboard images for offline calibration.

Shows the live camera, detects the board in every frame and saves the image
(without overlay) when SPACE is pressed. A green coverage map shows which areas
of the image already have a board: borders and corners must be covered too,
since that is where distortion is most noticeable.

Tips: 20-30 images; different distances; tilts of up to ~45° on both axes;
board still when saving (no motion blur); same resolution as the system
(config.yaml).

Usage:
    python calibration/capture.py [--camera 1]

Keys:
    SPACE    save image (only if the board is detected)
    z        undo: delete the last image saved in this session
    q / ESC  quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from calib_utils import find_corners, pattern_size  # noqa: E402
from camera import ThreadedCamera  # noqa: E402
from fps import FPSCounter, draw_stats  # noqa: E402
from settings import load_config, project_path  # noqa: E402

MIN_SHIFT_PX = 10.0  # an image almost identical to one already saved adds no information


def outer_quad(corners: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    """The 4 outer corners of the grid of detected corners."""
    cols, rows = pattern
    g = corners.reshape(rows, cols, 2)
    return np.array([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]], np.int32)


def add_coverage(coverage: np.ndarray, corners: np.ndarray, pattern: tuple[int, int]) -> None:
    """Add 1 over the area covered by the board (to see how many times each area has been covered)."""
    m = np.zeros(coverage.shape, np.uint8)
    cv2.fillConvexPoly(m, outer_quad(corners, pattern), 1)
    coverage += m


def main() -> None:
    p = argparse.ArgumentParser(description="Calibration image capture")
    p.add_argument("--camera", type=int, help="camera index (overrides the config)")
    p.add_argument("--out", type=Path, help="output folder (defaults to the one in the config)")
    args = p.parse_args()

    cfg = load_config()
    cam_cfg, cal = cfg["camera"], cfg["calibration"]
    pattern = pattern_size(cal["squares_x"], cal["squares_y"])
    out_dir = args.out or project_path(cal["images_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    index = args.camera if args.camera is not None else cam_cfg["index"]

    with ThreadedCamera(index, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"],
                        cam_cfg.get("autofocus"), cam_cfg.get("exposure")) as cam:
        w, h = cam.resolution
        if (w, h) != (cam_cfg["width"], cam_cfg["height"]):
            print(f"WARNING: the camera gives {w}x{h} and the config asks for "
                  f"{cam_cfg['width']}x{cam_cfg['height']}. Calibrate at the working resolution.")

        # Accumulated coverage (includes any images already in the folder)
        coverage = np.zeros((h, w), np.float32)
        saved: list[np.ndarray] = []
        existing = sorted(out_dir.glob("*.png"))
        for f in existing:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (h, w):
                continue
            c = find_corners(gray, pattern, subpix=False)
            if c is not None:
                add_coverage(coverage, c, pattern)
                saved.append(c.reshape(-1, 2))
        print(f"{len(existing)} existing images in {out_dir}. SPACE saves, z undoes, q quits.")

        session_files: list[Path] = []
        session_corners: list[np.ndarray] = []
        fps = FPSCounter()
        last_id, msg, msg_t = -1, "", 0.0
        window = "Calibration capture"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizing keeps the aspect ratio

        pending_key = 255  # key pressed while waiting for a new frame (applied to the next one)
        while True:
            frame_id, frame = cam.read()
            if frame_id == last_id:
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k != 255:
                    pending_key = k
                continue
            last_id = frame_id

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, pattern, fast=True, subpix=False)

            out = frame.copy()
            if coverage.any():
                tint = np.zeros_like(out)
                tint[..., 1] = np.clip(coverage * 60 + 60, 0, 255).astype(np.uint8)
                mask = coverage > 0
                out[mask] = cv2.addWeighted(out, 0.65, tint, 0.35, 0)[mask]
            if corners is not None:
                cv2.drawChessboardCorners(out, pattern, corners, True)
            pct = 100.0 * np.count_nonzero(coverage) / coverage.size
            n_total = len(saved)
            lines = [f"FPS: {fps.tick():.1f}", f"Images: {n_total}  Coverage: {pct:.0f}%",
                     "BOARD OK" if corners is not None else "no board"]
            if msg and time.time() - msg_t < 2.0:
                lines.append(msg)
            draw_stats(out, lines)
            cv2.imshow(window, out)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                key, pending_key = pending_key, 255
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if corners is None:
                    msg = "The whole board is not visible"
                else:
                    pts = corners.reshape(-1, 2)
                    shift = min((np.linalg.norm(pts - s, axis=1).mean() for s in saved), default=np.inf)
                    if shift < MIN_SHIFT_PX:
                        msg = "Almost the same as another one: move or tilt the board"
                    else:
                        name = out_dir / f"calib_{time.strftime('%Y%m%d_%H%M%S')}_{frame_id}.png"
                        cv2.imwrite(str(name), frame)  # clean image, no overlay
                        add_coverage(coverage, corners, pattern)
                        saved.append(pts)
                        session_files.append(name)
                        session_corners.append(pts)
                        msg = f"Saved {name.name}"
                        print(msg)
                msg_t = time.time()
            if key == ord("z") and session_files:
                f = session_files.pop()
                pts = session_corners.pop()
                f.unlink(missing_ok=True)
                saved = [s for s in saved if s is not pts]
                # Rebuild the coverage without the deleted image
                coverage[:] = 0
                for s in saved:
                    add_coverage(coverage, s, pattern)
                msg, msg_t = f"Deleted {f.name}", time.time()
                print(msg)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
