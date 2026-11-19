"""System entry point.

Reads the camera in real time and shows the FPS on every frame. If
calibration/calib.yaml exists, the 'u' key toggles distortion correction.
The card detector is on by default ('c' key).
The remaining modules (decoder, tracker, turret) will be plugged in here.

Usage:
    python src/main.py
    python src/main.py --camera 1

Keys:
    q / ESC  quit
    s        save the current frame to data/snapshots/
    u        toggle distortion correction
    c        toggle the card detector
    d        save the card detector stages to data/debug/ (report figure)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml

from calib_utils import _to_builtin
from camera import ThreadedCamera
from camera_model import CameraModel
from card_detector import CardDetector, draw_detections
from fps import FPSCounter, draw_stats
from settings import DEFAULT_CONFIG, ROOT, load_config, project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Virtual anti-drone turret")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--camera", type=int, help="camera index (overrides the config)")
    return p.parse_args()


def load_camera_model(cfg: dict, resolution) -> CameraModel | None:
    cal = cfg.get("calibration", {})
    path = project_path(cal.get("output", "calibration/calib.yaml"))
    if not path.exists():
        print(f"No calibration ({path}): run calibration/capture.py and calibration/calibrate.py")
        return None
    model = CameraModel.from_yaml(path, alpha=float(cal.get("alpha", 0.0)))
    if model.size != tuple(resolution):
        print(f"WARNING: calibrated at {model.size} but the camera runs at {resolution}: correction disabled")
        return None
    print(f"Calibration loaded from {path}. Press 'u' to correct distortion.")
    return model


def save_card_stages(dbg: dict, frame, out, dets=()) -> Path:
    """Save every detector stage as PNG (sequence of transformations for the report)
    and the features of each detected card in features.yaml (to debug failed reads)."""
    folder = ROOT / "data" / "debug" / f"cards_{time.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "0_original.png"), frame)
    for name, img in dbg.items():
        cv2.imwrite(str(folder / f"{name}.png"), img)
    cv2.imwrite(str(folder / "6_result.png"), out)
    feats = [{"label": d.label, "corners": d.quad.round(1).tolist(), **d.features} for d in dets]
    with open(folder / "features.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(feats), f, sort_keys=False, allow_unicode=True)
    return folder


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cam_cfg, disp_cfg = cfg["camera"], cfg["display"]
    index = args.camera if args.camera is not None else cam_cfg["index"]

    snapshot_dir = ROOT / disp_cfg["snapshot_dir"]
    window = disp_cfg["window_name"]
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps = FPSCounter()
    last_id = -1
    undistort = False
    detector = CardDetector.from_config(cfg)
    show_cards, save_debug = True, False

    with ThreadedCamera(index, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"],
                        cam_cfg.get("autofocus")) as cam:
        w, h = cam.resolution
        print(f"Camera {index} opened at {w}x{h}. Press 'q' to quit, 's' to capture.")
        model = load_camera_model(cfg, (w, h))

        while True:
            frame_id, frame = cam.read()
            if frame_id == last_id:  # no new frame yet: do not reprocess the same one
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue
            last_id = frame_id

            t0 = time.perf_counter()
            if cam_cfg.get("mirror", False):
                frame = cv2.flip(frame, 1)
            if undistort:
                frame = model.undistort(frame)
            dets, dbg = [], ({} if save_debug else None)
            if show_cards:
                dets = detector.detect(frame, dbg)
            # --- the remaining system modules will be plugged in here ---
            proc_ms = (time.perf_counter() - t0) * 1000

            out = frame.copy()
            lines = [f"FPS: {fps.tick():.1f}", f"Proc: {proc_ms:.1f} ms",
                     f"{w}x{h}  {'UNDISTORTED' if undistort else 'original'}"]
            if show_cards:
                draw_detections(out, dets, detector.roi(frame.shape))
                t = detector.timings
                lines.append(f"Cards: {t['frame_ms']:.1f} + {t['symbol_ms']:.1f} ms")
                lines += [f"  {d.label}" for d in dets]
            draw_stats(out, lines)
            cv2.imshow(window, out)
            if save_debug:
                print(f"Stages saved to {save_card_stages(dbg, frame, out, dets)}")
                save_debug = False

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                name = snapshot_dir / f"snap_{time.strftime('%Y%m%d_%H%M%S')}_{frame_id}.png"
                cv2.imwrite(str(name), frame)  # saved without overlay
                print(f"Saved {name}")
            if key == ord("c"):
                show_cards = not show_cards
            if key == ord("d"):
                if show_cards:
                    save_debug = True  # saved on the next frame
                else:
                    print("Enable the card detector with 'c' to save its stages")
            if key == ord("u"):
                if model is None:
                    print("No valid calibration to correct distortion")
                else:
                    undistort = not undistort

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
