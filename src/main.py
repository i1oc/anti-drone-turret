"""System entry point.

Reads the camera in real time and shows the FPS on every frame. If
calibration/calib.yaml exists, the 'u' key toggles distortion correction.
The system starts LOCKED: the card detector feeds the sequence decoder until the
right 4 cards are shown; then it is ARMED and the ball detector + multi-target
tracker run (bounding box, id, trail, speed and, if calibrated, angles/depth).

Usage:
    python src/main.py
    python src/main.py --camera 1

Keys:
    q / ESC  quit
    s        save the current frame to data/snapshots/
    u        toggle distortion correction
    c        toggle the card overlay (frames, labels and features)
    d        save the stages of the active detector (cards or balls) to data/debug/
    r        reset the sequence / disarm (does not shorten a penalty)
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import yaml

from calib_utils import _to_builtin
from camera import ThreadedCamera
from camera_model import CameraModel
from ball_detector import BallDetector
from card_detector import CardDetector, draw_detections
from decoder import ARMED, PENALTY, SequenceDecoder, select_label
from fps import FPSCounter, draw_stats
from hud import draw_decoder_panel, draw_tracks
from settings import DEFAULT_CONFIG, ROOT, load_config, project_path
from tracking import MultiTracker


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


def save_stages(prefix: str, dbg: dict, frame, out) -> Path:
    """Save the intermediate images of a detector (sequence of transformations for the report)."""
    folder = ROOT / "data" / "debug" / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "0_original.png"), frame)
    for name, img in dbg.items():
        cv2.imwrite(str(folder / f"{name}.png"), img)
    cv2.imwrite(str(folder / "9_result.png"), out)
    return folder


def save_decoder_log(dec: SequenceDecoder) -> Path | None:
    """Decoder events (reads, wrong codes, unlocks...) to CSV, for the unlock-time experiment."""
    if not dec.events:
        return None
    folder = ROOT / "data" / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"decoder_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "event", "detail"])
        for e in dec.events:
            wr.writerow([f"{e.t:.3f}", e.name, e.detail])
    return path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cam_cfg, disp_cfg = cfg["camera"], cfg["display"]
    index = args.camera if args.camera is not None else cam_cfg["index"]

    snapshot_dir = ROOT / disp_cfg["snapshot_dir"]
    window = disp_cfg["window_name"]
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizing keeps the aspect ratio

    fps = FPSCounter()
    last_id = -1
    undistort = False
    detector = CardDetector.from_config(cfg)
    decoder = SequenceDecoder.from_config(cfg)
    balls = BallDetector.from_config(cfg, ROOT)
    tracker = MultiTracker.from_config(cfg)
    tcfg = cfg.get("tracker", {})
    friends = tuple(tcfg.get("friend_colors", []))
    diameters = {name: (c.get("diameter_m") if isinstance(c, dict) else None) or tcfg.get("ball_diameter_m")
                 for name, c in tcfg.get("colors", {}).items()}
    show_cards, save_debug = True, False
    t_start = time.perf_counter()

    with ThreadedCamera(index, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"],
                        cam_cfg.get("autofocus"), cam_cfg.get("exposure")) as cam:
        w, h = cam.resolution
        print(f"Camera {index} opened at {w}x{h}. Press 'q' to quit, 's' to capture.")
        model = load_camera_model(cfg, (w, h))

        last_frame = None
        while True:
            # Single place where keys are read. waitKey(1) also refreshes the window.
            # (Before, keys pressed while waiting for a new camera frame were discarded.)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_frame is not None:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                name = snapshot_dir / f"snap_{time.strftime('%Y%m%d_%H%M%S')}_{last_id}.png"
                cv2.imwrite(str(name), last_frame)  # saved without overlay
                print(f"Saved {name}")
            elif key == ord("c"):
                show_cards = not show_cards
            elif key == ord("d"):
                if decoder.state != PENALTY:
                    save_debug = True  # saved on the next frame
                else:
                    print("Nothing runs during the penalty")
            elif key == ord("r"):
                decoder.reset(time.perf_counter() - t_start)
            elif key == ord("u"):
                if model is None:
                    print("No valid calibration to correct distortion")
                else:
                    undistort = not undistort

            frame_id, frame = cam.read()
            if frame_id == last_id:  # no new frame yet: do not reprocess the same one
                continue
            last_id = frame_id

            t0 = time.perf_counter()
            if cam_cfg.get("mirror", False):
                frame = cv2.flip(frame, 1)
            if undistort:
                frame = model.undistort(frame)
            last_frame = frame
            now = t0 - t_start
            dets, dbg = [], ({} if save_debug else None)
            # Only what the current mode needs runs: cards while locked, tracker when armed
            cards_on = decoder.state not in (ARMED, PENALTY)
            if cards_on:
                dets = detector.detect(frame, dbg)
                decoder.update(select_label([(d.label, d.area) for d in dets]), now)
            else:
                decoder.update(None, now)  # lets the penalty expire
            tracks = []
            if decoder.state == ARMED:
                t1 = time.perf_counter()
                found = balls.detect(frame, dbg)
                t2 = time.perf_counter()
                tracker.step(found, now)
                tracks = tracker.confirmed
                track_ms = (time.perf_counter() - t2) * 1000
                detect_ms = (t2 - t1) * 1000
            elif tracker.tracks:
                tracker.reset()  # disarmed: forget the targets
            proc_ms = (time.perf_counter() - t0) * 1000

            out = frame.copy()
            lines = [f"FPS {fps.tick():.1f}  |  proc {proc_ms:.1f} ms"]
            if cards_on:
                t = detector.timings
                lines.append(f"cards {t['frame_ms']:.1f} + {t['symbol_ms']:.1f} ms")
                if show_cards:
                    draw_detections(out, dets, detector.roi(frame.shape))
            if decoder.state == ARMED:
                lines.append(f"balls {detect_ms:.1f} ms  |  tracker {track_ms:.1f} ms  |  targets {len(tracks)}")
                draw_tracks(out, tracks, friends, model, undistort, diameters)
            if undistort:
                lines.append("UNDISTORTED")
            draw_stats(out, lines)
            draw_decoder_panel(out, decoder, now)
            cv2.imshow(window, out)
            if save_debug:
                if decoder.state == ARMED:
                    print(f"Stages saved to {save_stages('balls', dbg, frame, out)}")
                elif cards_on:
                    print(f"Stages saved to {save_card_stages(dbg, frame, out, dets)}")
                save_debug = False

    cv2.destroyAllWindows()
    log = save_decoder_log(decoder)
    if log:
        print(f"Decoder events saved to {log}")


if __name__ == "__main__":
    main()
