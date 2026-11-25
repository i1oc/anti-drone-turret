"""Tune the HSV colour ranges of the balls by clicking on them.

Left: live image with the balls detected using the current ranges.
Right: mask of the selected colour (white = pixels accepted).

Select a colour with its number key, click several times on the ball of that
colour (lit side, shadow side, different positions of the swing); each click
samples a small disc of pixels and the range is recomputed from all samples.
For the basketball click on the ORANGE surface, not on the black seams (the
closing fills them). Colours with motion_confirm only appear while moving.
Press 'w' to save config/colors.yaml, which overrides the defaults of config.yaml.

Usage:
    python tools/pick_colors.py [--camera 1]

Keys:
    1..9     select colour (order of tracker.colors in config.yaml)
    click    add samples of the selected colour
    x        clear the samples of the selected colour (back to the default range)
    w        write config/colors.yaml
    q / ESC  quit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from ball_detector import BallDetector, hsv_ranges_from_samples  # noqa: E402
from camera import ThreadedCamera  # noqa: E402
from fps import draw_stats  # noqa: E402
from settings import ROOT, load_config  # noqa: E402

SAMPLE_RADIUS = 6  # px around the click


def main() -> None:
    p = argparse.ArgumentParser(description="Tune ball colours by clicking on them")
    p.add_argument("--camera", type=int, help="camera index (overrides the config)")
    args = p.parse_args()

    cfg = load_config()
    cam_cfg, tcfg = cfg["camera"], cfg["tracker"]
    det = BallDetector.from_config(cfg, ROOT)
    names = list(det.colors)
    tuned: dict[str, list] = {}
    samples: dict[str, list[np.ndarray]] = {n: [] for n in names}
    current = 0
    state = {"hsv": None, "clicks": []}
    out_file = ROOT / tcfg.get("colors_file", "config/colors.yaml")
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            tuned = yaml.safe_load(f) or {}
        print(f"Loaded tuned ranges from {out_file}: {list(tuned)}")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["hsv"] is not None and x < state["hsv"].shape[1]:
            state["clicks"].append((x, y))

    window = "Pick colours"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizing keeps the aspect ratio
    cv2.setMouseCallback(window, on_mouse)
    index = args.camera if args.camera is not None else cam_cfg["index"]
    with ThreadedCamera(index, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"], cam_cfg.get("autofocus"), cam_cfg.get("exposure")) as cam:
        last_id = -1
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("9") and key - ord("1") < len(names):
                current = key - ord("1")
            name = names[current]
            if key == ord("x"):
                samples[name] = []
                tuned.pop(name, None)
                det.set_ranges(name, cfg["tracker"]["colors"][name]["ranges"]
                               if isinstance(cfg["tracker"]["colors"][name], dict) else cfg["tracker"]["colors"][name])
            if key == ord("w"):
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write("# HSV ranges tuned with tools/pick_colors.py (override config.yaml)\n")
                    yaml.safe_dump(tuned, f, default_flow_style=None, sort_keys=False)
                print(f"Saved {out_file}: {tuned}")

            frame_id, frame = cam.read()
            if frame_id == last_id:
                continue
            last_id = frame_id
            blurred = cv2.GaussianBlur(frame, (det.blur, det.blur), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            state["hsv"] = hsv

            if state["clicks"]:
                m = np.zeros(hsv.shape[:2], np.uint8)
                for (x, y) in state["clicks"]:
                    cv2.circle(m, (x, y), SAMPLE_RADIUS, 255, -1)
                state["clicks"] = []
                samples[name].append(hsv[m > 0])
                ranges = hsv_ranges_from_samples(np.concatenate(samples[name]))
                tuned[name] = ranges
                det.set_ranges(name, ranges)
                print(f"{name}: {ranges}")

            dets = det.detect(frame)
            out = frame.copy()
            for d in dets:
                cv2.circle(out, (int(d.x), int(d.y)), int(d.r), (255, 255, 255), 2)
                cv2.putText(out, d.color, (int(d.x - d.r), int(d.y - d.r) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            mask = det.color_mask(hsv, name)
            keys = "  ".join(f"{i + 1}:{n}{'*' if n in tuned else ''}" for i, n in enumerate(names))
            draw_stats(out, [keys, f"selected: {name} ({sum(len(s) for s in samples[name])} px sampled)",
                             "click = sample, x = clear, w = save, q = quit"], corner="top-left")
            cv2.imshow(window, np.hstack([out, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
