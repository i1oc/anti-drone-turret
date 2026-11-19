"""Offline camera calibration from the chessboard images (Zhang's method).

Steps:
  1. findChessboardCorners + cornerSubPix on every image.
  2. Reproducible training / validation split (config: val_fraction, seed).
  3. calibrateCamera with two distortion models: 5 coefficients (k1, k2, p1, p2, k3)
     and 4 coefficients (k3 = 0). The calibration RMS always drops when parameters
     are added; the VALIDATION RMS detects overfitting. The simplest model is
     chosen unless the more complex one improves validation by at least 2 %.
  4. Saves calibration/calib.yaml and the report figures to docs/figures/calibration/.

Usage:
    python calibration/calibrate.py
    python calibration/calibrate.py --images other/folder --no-curve
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from calib_utils import (board_points, calibrate, find_corners, pattern_size,  # noqa: E402
                         reprojection_residuals, rms_of, save_calibration, split_train_val,
                         validation_residuals)
from camera_model import CameraModel  # noqa: E402
from settings import load_config, project_path  # noqa: E402

MODELS = [  # from simplest to most complex
    ("4 coeffs (k3 = 0)", cv2.CALIB_FIX_K3),
    ("5 coeffs (k1 k2 p1 p2 k3)", 0),
]
MIN_GAIN = 0.02  # minimum relative improvement of the validation RMS to accept a more complex model
STD_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]

# Categorical palette (colour-blind safe): training = blue, validation = orange
C_TRAIN, C_VAL = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e4e4e4"


# ----------------------------------------------------------------------------- analysis

def error_vs_num_images(obj, img, size, flags, val_obj, val_img, n_values, repeats=10, seed=0):
    """Calibration and validation RMS versus number of images (question A.3 of lab 2).

    For each n, calibrate with n randomly chosen training images and always
    validate on the same validation set.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        cal, val = [], []
        for _ in range(repeats):
            idx = rng.choice(len(img), size=n, replace=False)
            r = calibrate([obj[i] for i in idx], [img[i] for i in idx], size, flags)
            cal.append(r["rms"])
            val.append(rms_of(validation_residuals(val_obj, val_img, r["K"], r["dist"])))
        rows.append((n, np.mean(cal), np.std(cal), np.mean(val), np.std(val)))
    return np.array(rows)


# ----------------------------------------------------------------------------- figures

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_corners(paths, corners, pattern, out):
    """Up to 4 examples of corner detection."""
    sel = np.linspace(0, len(paths) - 1, min(4, len(paths))).astype(int)
    fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 3.2))
    for ax, i in zip(np.atleast_1d(axes), sel):
        im = cv2.imread(str(paths[i]))
        cv2.drawChessboardCorners(im, pattern, corners[i], True)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(paths[i].name, fontsize=8, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_per_image_error(names_tr, err_tr, names_va, err_va, rms, val_rms, out):
    names = list(names_tr) + list(names_va)
    err = np.concatenate([err_tr, err_va])
    colors = [C_TRAIN] * len(err_tr) + [C_VAL] * len(err_va)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(names) + 2), 3.6))
    _style(ax)
    ax.bar(x, err, color=colors, width=0.7)
    l1 = ax.axhline(rms, color=C_TRAIN, lw=1.5, ls="--", label=f"Calibration RMS {rms:.3f} px")
    l2 = ax.axhline(val_rms, color=C_VAL, lw=1.5, ls=":", label=f"Validation RMS {val_rms:.3f} px")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(len(names))], fontsize=7)
    ax.set_xlabel("Image (index in calib.yaml)", color=INK)
    ax.set_ylabel("RMS reprojection error [px]", color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_TRAIN, label="Training"), Patch(color=C_VAL, label="Validation"),
                       l1, l2], frameon=False, fontsize=8, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_residuals(res_tr, res_va, out):
    """Residual cloud (projection − detection). It should be round and centred at 0."""
    rt, rv = np.concatenate(res_tr), np.concatenate(res_va)
    lim = max(0.6, np.abs(np.concatenate([rt, rv])).max() * 1.1)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    _style(ax)
    ax.scatter(rt[:, 0], rt[:, 1], s=6, color=C_TRAIN, alpha=0.5, marker="o", lw=0, label="Training")
    ax.scatter(rv[:, 0], rv[:, 1], s=10, color=C_VAL, alpha=0.7, marker="^", lw=0, label="Validation")
    ax.add_patch(plt.Circle((0, 0), 0.5, fill=False, color=MUTED, ls="--", lw=1))
    ax.text(0.36, 0.36, "0.5 px", color=MUTED, fontsize=8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Δu [px]", color=INK)
    ax.set_ylabel("Δv [px]", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_coverage(corners_tr, corners_va, size, out):
    """Position of every corner used: they should cover the whole image, borders included."""
    w, h = size
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    _style(ax)
    ax.grid(False)
    for c in corners_tr:
        p = c.reshape(-1, 2)
        ax.plot(p[:, 0], p[:, 1], ".", ms=2.5, color=C_TRAIN)
    for c in corners_va:
        p = c.reshape(-1, 2)
        ax.plot(p[:, 0], p[:, 1], "^", ms=3, color=C_VAL)
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color=MUTED, lw=1)
    ax.set_xlim(-10, w + 10)
    ax.set_ylim(h + 10, -10)
    ax.set_aspect("equal")
    ax.set_xlabel("u [px]", color=INK)
    ax.set_ylabel("v [px]", color=INK)
    ax.plot([], [], ".", color=C_TRAIN, label="Training")
    ax.plot([], [], "^", color=C_VAL, label="Validation")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_distortion_field(K, dist, size, out):
    """Displacement introduced by the lens: |distorted pixel − ideal| at each point."""
    w, h = size
    u, v = np.meshgrid(np.linspace(0, w - 1, 64), np.linspace(0, h - 1, 48))
    pix = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pix, K, dist, P=K).reshape(-1, 2)
    mag = np.linalg.norm(pix.reshape(-1, 2) - ideal, axis=1).reshape(u.shape)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.set_aspect("equal")
    cs = ax.contourf(u, v, mag, levels=12, cmap="Blues")
    su, sv = np.meshgrid(np.linspace(20, w - 20, 16), np.linspace(20, h - 20, 12))
    sp = np.stack([su.ravel(), sv.ravel()], 1).reshape(-1, 1, 2)
    si = cv2.undistortPoints(sp, K, dist, P=K).reshape(-1, 2)
    d = sp.reshape(-1, 2) - si
    ax.quiver(si[:, 0], si[:, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1,
              color=INK, width=0.003)
    ax.plot(K[0, 2], K[1, 2], "+", color=C_VAL, ms=10, mew=2)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("u [px]", color=INK)
    ax.set_ylabel("v [px]", color=INK)
    fig.colorbar(cs, ax=ax, label="Distortion displacement [px]", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_undistort(path, model, out):
    """Original and undistorted image side by side."""
    im = cv2.imread(str(path))
    und = model.undistort(im)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, x, t in zip(axes, (im, und), ("Original", "Undistorted (remap)")):
        ax.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        ax.set_title(t, color=INK, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_extrinsics(obj, rvecs, tvecs, val_rvecs, val_tvecs, out):
    """Board positions relative to the camera: shows the variety of poses."""
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(projection="3d")
    cols, rows = obj[:, 0].max(), obj[:, 1].max()
    quad = np.array([[0, 0, 0], [cols, 0, 0], [cols, rows, 0], [0, rows, 0], [0, 0, 0]], float)
    for rv, tv, c in [(r, t, C_TRAIN) for r, t in zip(rvecs, tvecs)] + \
                     [(r, t, C_VAL) for r, t in zip(val_rvecs, val_tvecs)]:
        R, _ = cv2.Rodrigues(rv)
        p = quad @ R.T + np.ravel(tv)
        # plot axes: X right, Z (depth) into the page, -Y up
        ax.plot(p[:, 0], p[:, 2], -p[:, 1], color=c, lw=1.2)
    s = 0.15 * np.max(np.abs(np.concatenate([np.ravel(t) for t in list(tvecs) + list(val_tvecs)])))
    # Camera as a pyramid: apex at the optical centre, base at Z = s
    base = np.array([[-s, -.75 * s, s], [s, -.75 * s, s], [s, .75 * s, s], [-s, .75 * s, s], [-s, -.75 * s, s]])
    ax.plot(base[:, 0], base[:, 2], -base[:, 1], color=INK, lw=1)
    for b in base[:4]:
        ax.plot([0, b[0]], [0, b[2]], [0, -b[1]], color=INK, lw=1)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_zlabel("−Y [mm]")
    ax.plot([], [], color=C_TRAIN, label="Training")
    ax.plot([], [], color=C_VAL, label="Validation")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_error_vs_n(rows, out):
    n, cm, cs, vm, vs = rows.T
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _style(ax)
    ax.errorbar(n, cm, yerr=cs, fmt="o-", color=C_TRAIN, ms=5, lw=2, capsize=3, label="Calibration")
    ax.errorbar(n, vm, yerr=vs, fmt="^-", color=C_VAL, ms=6, lw=2, capsize=3, label="Validation")
    ax.set_xlabel("Number of calibration images", color=INK)
    ax.set_ylabel("RMS reprojection error [px]", color=INK)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="Offline camera calibration")
    p.add_argument("--images", type=Path, help="image folder (defaults to the one in the config)")
    p.add_argument("--output", type=Path, help="output YAML (defaults to the one in the config)")
    p.add_argument("--figures", type=Path, help="figures folder (defaults to the one in the config)")
    p.add_argument("--no-figures", action="store_true", help="do not generate figures")
    p.add_argument("--no-curve", action="store_true", help="skip the error vs number of images curve")
    args = p.parse_args()

    cfg = load_config()
    cal = cfg["calibration"]
    pattern = pattern_size(cal["squares_x"], cal["squares_y"])
    square = float(cal["square_mm"])
    obj = board_points(pattern, square)
    img_dir = args.images or project_path(cal["images_dir"])
    out_yaml = args.output or project_path(cal["output"])
    fig_dir = args.figures or project_path(cal["figures_dir"])

    paths = sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg")])
    if not paths:
        raise SystemExit(f"No images in {img_dir}. Capture them with calibration/capture.py")

    # 1. Corner detection
    size, ok_paths, corners, failed = None, [], [], []
    for f in paths:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        s = gray.shape[::-1]
        if size is None:
            size = s
        elif s != size:
            raise SystemExit(f"{f.name} is {s[0]}x{s[1]} and the others {size[0]}x{size[1]}")
        c = find_corners(gray, pattern)
        if c is None:
            failed.append(f.name)
        else:
            ok_paths.append(f)
            corners.append(c)
    print(f"Board {cal['squares_x']}x{cal['squares_y']} squares ({pattern[0]}x{pattern[1]} corners), "
          f"{square:g} mm. Images {size[0]}x{size[1]}")
    print(f"Detected in {len(ok_paths)}/{len(paths)} images" + (f". Failed: {failed}" if failed else ""))
    expected = (cfg["camera"]["width"], cfg["camera"]["height"])
    if size != expected:
        print(f"WARNING: the images are {size} and the system runs at {expected}: the calibration will not be valid.")
    if len(ok_paths) < 6:
        raise SystemExit("At least 6 valid images are needed (20-30 recommended).")

    # 2. Training / validation split
    tr, va = split_train_val(len(ok_paths), float(cal["val_fraction"]), int(cal["seed"]))
    obj_tr, img_tr = [obj] * len(tr), [corners[i] for i in tr]
    obj_va, img_va = [obj] * len(va), [corners[i] for i in va]
    print(f"Training: {len(tr)} images · Validation: {len(va)} images")

    # 3. Calibration with each distortion model
    results = []
    print(f"\n{'Model':<26}{'Calib. RMS [px]':>17}{'Valid. RMS [px]':>17}")
    for name, flags in MODELS:
        r = calibrate(obj_tr, img_tr, size, flags)
        r["val_res"] = validation_residuals(obj_va, img_va, r["K"], r["dist"]) if len(va) else []
        r["val_rms"] = rms_of(r["val_res"]) if len(va) else float("nan")
        r["name"], r["flags"] = name, flags
        results.append(r)
        print(f"{name:<26}{r['rms']:>17.4f}{r['val_rms']:>17.4f}")
    best = results[0]
    for r in results[1:]:
        if len(va) and r["val_rms"] < (1 - MIN_GAIN) * best["val_rms"]:
            best = r
    K, dist = best["K"], best["dist"]
    print(f"Chosen model: {best['name']} (the complex one only if it improves validation by > {MIN_GAIN:.0%})")

    # Summary
    std = best["std_intrinsics"]
    print("\nK =\n", np.array2string(K, precision=2, suppress_small=True))
    for i, n in enumerate(STD_NAMES[:4]):
        v = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]][i]
        print(f"  {n} = {v:8.2f} ± {std[i]:.2f} px")
    print("dist (k1 k2 p1 p2 k3) =", np.array2string(dist[:5], precision=4))
    print(f"Calibration RMS = {best['rms']:.4f} px · Validation RMS = {best['val_rms']:.4f} px")
    fov_h = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    fov_v = 2 * np.degrees(np.arctan(size[1] / (2 * K[1, 1])))
    print(f"Approx. field of view {fov_h:.1f}° x {fov_v:.1f}°")

    # Per-image error and possible bad images
    err_tr = best["per_view_rms"]
    err_va = np.array([rms_of([r]) for r in best["val_res"]])
    names_tr = [ok_paths[i].name for i in tr]
    names_va = [ok_paths[i].name for i in va]
    all_err = np.concatenate([err_tr, err_va])
    thr = max(1.0, all_err.mean() + 2 * all_err.std())
    bad = [(n, e) for n, e in zip(names_tr + names_va, all_err) if e > thr]
    if bad:
        print(f"\nImages with high error (> {thr:.2f} px): check whether they are blurred or badly detected")
        for n, e in bad:
            print(f"  {n}: {e:.3f} px")

    save_calibration(
        out_yaml, K, dist, size,
        rms=round(best["rms"], 5),
        val_rms=round(best["val_rms"], 5) if len(va) else None,
        model=best["name"],
        std_fx_fy_cx_cy=[round(float(x), 4) for x in std[:4]],
        fov_deg=[round(fov_h, 2), round(fov_v, 2)],
        board={"squares_x": cal["squares_x"], "squares_y": cal["squares_y"], "square_mm": square},
        models={r["name"]: {"rms": round(r["rms"], 5), "val_rms": round(r["val_rms"], 5)} for r in results},
        train_images={n: round(float(e), 4) for n, e in zip(names_tr, err_tr)},
        val_images={n: round(float(e), 4) for n, e in zip(names_va, err_va)},
        failed_images=failed,
    )
    print(f"\nSaved {out_yaml}")

    if args.no_figures:
        return

    # 4. Report figures
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_tr = [reprojection_residuals(obj, img_tr[k], best["rvecs"][k], best["tvecs"][k], K, dist)
              for k in range(len(tr))]
    model = CameraModel(K, dist, size, float(cal.get("alpha", 0.0)))
    val_poses = [cv2.solvePnP(obj, c, K, dist)[1:] for c in img_va]
    # Image for the comparison: the one with the board farthest from the centre (most distortion)
    far = int(np.argmax([np.linalg.norm(c.reshape(-1, 2).mean(0) - K[:2, 2]) for c in corners]))

    fig_corners(ok_paths, corners, pattern, fig_dir / "detected_corners.png")
    fig_per_image_error(names_tr, err_tr, names_va, err_va, best["rms"], best["val_rms"],
                        fig_dir / "per_image_error.png")
    fig_residuals(res_tr, best["val_res"], fig_dir / "residuals.png")
    fig_coverage(img_tr, img_va, size, fig_dir / "coverage.png")
    fig_distortion_field(K, dist, size, fig_dir / "distortion_field.png")
    fig_undistort(ok_paths[far], model, fig_dir / "original_vs_undistorted.png")
    fig_extrinsics(obj, best["rvecs"], best["tvecs"], [p[0] for p in val_poses],
                   [p[1] for p in val_poses], fig_dir / "board_poses.png")
    if not args.no_curve and len(tr) >= 6 and len(va):
        n_values = list(range(3, len(tr) + 1, max(1, len(tr) // 10)))
        rows = error_vs_num_images(obj_tr, img_tr, size, best["flags"], obj_va, img_va, n_values)
        fig_error_vs_n(rows, fig_dir / "error_vs_num_images.png")
    print(f"Figures in {fig_dir}")


if __name__ == "__main__":
    main()
