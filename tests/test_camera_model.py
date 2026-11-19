"""Camera model tests: angles, distortion correction and depth."""
import cv2
import numpy as np
import pytest

import synth
from camera_model import CameraModel, angles_from_normalized, normalized_from_angles


@pytest.fixture(scope="module")
def cam():
    return CameraModel(synth.K_TRUE, synth.DIST_TRUE, synth.SIZE)


def test_angles_basic_directions():
    az, el = angles_from_normalized(0.0, 0.0)
    assert az == 0 and el == 0
    az, el = angles_from_normalized(1.0, 0.0)
    assert np.degrees(az) == pytest.approx(45)     # right = positive azimuth
    az, el = angles_from_normalized(0.0, -1.0)
    assert np.degrees(el) == pytest.approx(45)     # up (negative y) = positive elevation
    _, el = angles_from_normalized(1.0, 1.0)
    assert np.degrees(el) == pytest.approx(-np.degrees(np.arctan(1 / np.sqrt(2))))


def test_angles_inverse_roundtrip():
    rng = np.random.default_rng(0)
    az, el = rng.uniform(-0.6, 0.6, 50), rng.uniform(-0.5, 0.5, 50)
    x, y = normalized_from_angles(az, el)
    az2, el2 = angles_from_normalized(x, y)
    np.testing.assert_allclose(az2, az, atol=1e-12)
    np.testing.assert_allclose(el2, el, atol=1e-12)


def test_principal_point_is_optical_axis(cam):
    az, el = cam.pixel_to_angles([[318.0, 243.0]])
    assert abs(az[0]) < 1e-9 and abs(el[0]) < 1e-9


def test_pixel_to_angles_recovers_known_directions(cam):
    """A ray of known direction is projected with distortion and its angle is recovered."""
    rng = np.random.default_rng(1)
    az, el = rng.uniform(-0.4, 0.4, 30), rng.uniform(-0.3, 0.3, 30)
    x, y = normalized_from_angles(az, el)
    pts3d = np.column_stack([x, y, np.ones_like(x)])
    pix, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), synth.K_TRUE, synth.DIST_TRUE)
    az2, el2 = cam.pixel_to_angles(pix.reshape(-1, 2))
    assert np.degrees(np.abs(az2 - az)).max() < 0.01
    assert np.degrees(np.abs(el2 - el)).max() < 0.01


def test_angles_same_in_original_and_undistorted_image(cam):
    pix = np.array([[50.0, 40.0], [600.0, 420.0], [320.0, 100.0]])
    az1, el1 = cam.pixel_to_angles(pix)
    az2, el2 = cam.pixel_to_angles(cam.undistort_points(pix), undistorted=True)
    np.testing.assert_allclose(az2, az1, atol=1e-6)
    np.testing.assert_allclose(el2, el1, atol=1e-6)


def test_undistort_straightens_board_lines(cam):
    """After correction, the rows of board corners must be straight."""
    rvec, tvec = synth.random_pose(np.random.default_rng(4))
    from calib_utils import board_points, pattern_size
    obj = board_points(pattern_size(*synth.SQUARES), synth.SQUARE_MM)
    pix, _ = cv2.projectPoints(obj, rvec, tvec, synth.K_TRUE, synth.DIST_TRUE)

    def max_line_dev(p):
        dev = 0.0
        for row in p.reshape(7, 10, 2):
            c = row - row.mean(0)
            _, s, _ = np.linalg.svd(c)
            dev = max(dev, s[1] / np.sqrt(len(row)))   # deviation from the fitted line
        return dev

    assert max_line_dev(cam.undistort_points(pix.reshape(-1, 2))) < 0.01 < max_line_dev(pix)


def test_undistort_frame_size(cam):
    out = cam.undistort(np.zeros((480, 640, 3), np.uint8))
    assert out.shape == (480, 640, 3)
    with pytest.raises(ValueError):
        cam.undistort(np.zeros((720, 1280, 3), np.uint8))


def test_depth_from_diameter(cam):
    f = cam.focal
    assert cam.depth_from_diameter(f * 0.07 / 1.5, 0.07) == pytest.approx(1.5)
