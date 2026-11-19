"""Calibration tests with a virtual camera of known parameters."""
import cv2
import numpy as np
import pytest

import synth
from calib_utils import (board_points, calibrate, find_corners, load_calibration, pattern_size,
                         reprojection_residuals, rms_of, save_calibration, split_train_val,
                         validation_residuals)

PATTERN = pattern_size(*synth.SQUARES)
OBJ = board_points(PATTERN, synth.SQUARE_MM)


def synthetic_views(n, noise_px=0.1, seed=0):
    """n views: 3D points, projections with Gaussian noise and true poses."""
    rng = np.random.default_rng(seed)
    obj_pts, img_pts = [], []
    for _ in range(n):
        rvec, tvec = synth.random_pose(rng)
        proj, _ = cv2.projectPoints(OBJ, rvec, tvec, synth.K_TRUE, synth.DIST_TRUE)
        proj += rng.normal(0, noise_px, proj.shape)
        obj_pts.append(OBJ)
        img_pts.append(proj.astype(np.float32))
    return obj_pts, img_pts


def test_pattern_size_from_squares():
    assert pattern_size(11, 8) == (10, 7)


def test_board_points_follow_opencv_order():
    assert OBJ.shape == (70, 3)
    np.testing.assert_allclose(OBJ[0], [0, 0, 0])
    np.testing.assert_allclose(OBJ[1], [20, 0, 0])     # X advances first (along the row)
    np.testing.assert_allclose(OBJ[10], [0, 20, 0])    # the next row
    np.testing.assert_allclose(OBJ[-1], [180, 120, 0])


def test_calibration_recovers_intrinsics():
    obj, img = synthetic_views(15, noise_px=0.1)
    res = calibrate(obj, img, synth.SIZE)
    K = res["K"]
    assert res["rms"] < 0.2
    assert abs(K[0, 0] - 620) / 620 < 0.01
    assert abs(K[1, 1] - 615) / 615 < 0.01
    assert abs(K[0, 2] - 318) < 3 and abs(K[1, 2] - 243) < 3
    assert abs(res["dist"][0] - synth.DIST_TRUE[0]) < 0.03


def test_per_view_rms_matches_own_residuals():
    obj, img = synthetic_views(8, noise_px=0.2, seed=1)
    res = calibrate(obj, img, synth.SIZE)
    own = [rms_of([reprojection_residuals(o, i, r, t, res["K"], res["dist"])])
           for o, i, r, t in zip(obj, img, res["rvecs"], res["tvecs"])]
    np.testing.assert_allclose(own, res["per_view_rms"], atol=1e-3)
    # OpenCV's global RMS is the one over all points together
    all_res = [reprojection_residuals(o, i, r, t, res["K"], res["dist"])
               for o, i, r, t in zip(obj, img, res["rvecs"], res["tvecs"])]
    assert rms_of(all_res) == pytest.approx(res["rms"], abs=1e-3)


def test_validation_rms_on_unseen_views():
    obj, img = synthetic_views(18, noise_px=0.1, seed=2)
    res = calibrate(obj[:13], img[:13], synth.SIZE)
    val = rms_of(validation_residuals(obj[13:], img[13:], res["K"], res["dist"]))
    assert val < 0.25


def test_split_train_val():
    tr, va = split_train_val(20, 0.2, seed=3)
    assert len(va) == 4 and len(tr) == 16
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(range(20))
    tr2, va2 = split_train_val(20, 0.2, seed=3)
    assert (tr == tr2).all() and (va == va2).all()
    tr, va = split_train_val(3, 0.1)
    assert len(va) == 1 and len(tr) == 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_find_corners_on_rendered_image(seed):
    rng = np.random.default_rng(seed)
    rvec, tvec = synth.random_pose(rng)
    gray = synth.render_view(rvec, tvec, rng=rng)
    corners = find_corners(gray, PATTERN)
    assert corners is not None and corners.shape == (70, 1, 2)
    truth, _ = cv2.projectPoints(OBJ, rvec, tvec, synth.K_TRUE, synth.DIST_TRUE)
    err = np.linalg.norm(corners.reshape(-1, 2) - truth.reshape(-1, 2), axis=1)
    assert err.mean() < 0.25 and err.max() < 0.8   # same order and subpixel accuracy


def test_find_corners_returns_none_without_board():
    gray = np.full((480, 640), 128, np.uint8)
    assert find_corners(gray, PATTERN, fast=True) is None


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "calib.yaml"
    save_calibration(path, synth.K_TRUE, synth.DIST_TRUE, synth.SIZE, rms=0.31, n_train=10)
    c = load_calibration(path)
    np.testing.assert_allclose(c["K"], synth.K_TRUE)
    np.testing.assert_allclose(c["dist"], synth.DIST_TRUE)
    assert c["image_size"] == synth.SIZE and c["rms"] == 0.31 and c["n_train"] == 10
