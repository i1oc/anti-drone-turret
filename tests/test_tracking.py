"""Tests of the tracking core (Kalman, association, track management) with simulated targets."""
import cv2
import numpy as np
import pytest

from tracking import (CONFIRMED, LOST, TENTATIVE, Detection, KalmanFilter, MultiTracker, associate)

DT = 1 / 30


# ------------------------------------------------------------------ Kalman
def test_cv_filter_learns_velocity():
    rng = np.random.default_rng(0)
    kf = KalmanFilter("cv", 100, 200, q=50.0 ** 2, r=1.0)
    v = np.array([120.0, -60.0])
    for k in range(1, 60):
        kf.predict(DT)
        kf.update(np.array([100, 200]) + v * k * DT + rng.normal(0, 1.0, 2))
    np.testing.assert_allclose(kf.velocity, v, rtol=0.05, atol=3)
    # 0.5 s ahead, along a straight line
    true_ahead = np.array([100, 200]) + v * (59 * DT + 0.5)
    assert np.linalg.norm(kf.predict_ahead(0.5) - true_ahead) < 5


def test_ca_filter_learns_acceleration():
    kf = KalmanFilter("ca", 0, 0, q=10.0 ** 2, r=0.5)
    a = np.array([300.0, 100.0])
    for k in range(1, 90):
        kf.predict(DT)
        kf.update(0.5 * a * (k * DT) ** 2)
    np.testing.assert_allclose(kf.x[4:6], a, rtol=0.05)


def test_matches_opencv_kalman_filter():
    """Same model and noises as cv2.KalmanFilter: identical estimates (validates the equations)."""
    q, r = 400.0 ** 2, 2.0
    ours = KalmanFilter("cv", 50, 60, q=q, r=r)
    F, Q = ours.matrices(DT)
    ocv = cv2.KalmanFilter(4, 2)
    ocv.transitionMatrix = F.astype(np.float32)
    ocv.processNoiseCov = Q.astype(np.float32)
    ocv.measurementMatrix = ours.H.astype(np.float32)
    ocv.measurementNoiseCov = ours.R.astype(np.float32)
    ocv.statePost = ours.x.reshape(4, 1).astype(np.float32)
    ocv.errorCovPost = ours.P.astype(np.float32)
    rng = np.random.default_rng(1)
    for k in range(40):
        z = np.array([50 + 3 * k, 60 - 2 * k]) + rng.normal(0, 2, 2)
        ours.predict(DT)
        ocv.predict()
        ours.update(z)
        ocv.correct(z.reshape(2, 1).astype(np.float32))
    np.testing.assert_allclose(ours.x, ocv.statePost.ravel(), rtol=1e-3, atol=1e-2)


def test_mahalanobis_grows_with_distance_and_shrinks_with_uncertainty():
    kf = KalmanFilter("cv", 0, 0, q=1.0, r=1.0)
    assert kf.mahalanobis2((1, 0)) < kf.mahalanobis2((5, 0))
    d_before = kf.mahalanobis2((20, 0))
    for _ in range(10):
        kf.predict(DT)          # no updates: uncertainty grows
    assert kf.mahalanobis2((20, 0)) < d_before


# ------------------------------------------------------------------ association
def test_associate_optimal_and_gated():
    cost = np.array([[1.0, 2.0, 50.0],
                     [0.5, 3.0, np.inf]])
    matches, ut, ud = associate(cost, gate=5.991)
    # greedy would give track 0 -> det 0; the optimum is 0->1, 1->0 (total 2.5 < 4.0)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ut == [] and ud == [2]


def test_associate_empty():
    assert associate(np.zeros((0, 3))) == ([], [], [0, 1, 2])
    assert associate(np.zeros((2, 0))) == ([], [0, 1], [])


# ------------------------------------------------------------------ track management
def run(tracker, frames, t0=0.0):
    t = t0
    for dets in frames:
        tracker.step(dets, t)
        t += DT
    return t


def test_track_life_cycle():
    tr = MultiTracker(confirm_hits=3, max_misses=10)
    frames = [[Detection(100 + 4 * k, 100, 15, "red")] for k in range(3)]
    t = run(tr, frames[:2])
    assert tr.tracks[0].state == TENTATIVE and tr.confirmed == []
    t = run(tr, frames[2:], t)
    track = tr.tracks[0]
    assert track.state == CONFIRMED and track.id == 1
    # Disappears: coasts along its velocity, then is deleted
    x_before = track.position[0]
    t = run(tr, [[]] * 5, t)
    assert track.state == LOST and track.position[0] > x_before
    t = run(tr, [[]] * 10, t)
    assert tr.tracks == []
    # A new target gets a new id
    run(tr, [[Detection(300, 300, 15, "red")]] * 3, t)
    assert tr.confirmed[0].id == 2


def test_single_false_detection_does_not_consume_an_id():
    tr = MultiTracker()
    t = run(tr, [[Detection(50, 50, 10, "blue")]] + [[]] * 3)
    assert tr.tracks == []
    run(tr, [[Detection(300, 300, 15, "red")]] * 3, t)
    assert tr.confirmed[0].id == 1


def test_lost_track_is_recovered_with_same_id():
    tr = MultiTracker()
    t = 0.0
    for k in range(20):
        x = 100 + 300 * t
        dets = [] if 8 <= k < 13 else [Detection(x, 200, 15, "red")]  # 5 frames occluded
        tr.step(dets, t)
        t += DT
    assert len(tr.confirmed) == 1 and tr.confirmed[0].id == 1 and tr.confirmed[0].state == CONFIRMED


def test_colours_are_never_mixed():
    tr = MultiTracker()
    t = 0.0
    for k in range(30):  # red goes right, blue goes left, they cross at k = 15 at the same point
        dets = [Detection(100 + 10 * k, 200, 15, "red"), Detection(400 - 10 * k, 200, 15, "blue")]
        tr.step(dets, t)
        t += DT
    by_color = {trk.color: trk for trk in tr.confirmed}
    assert by_color["red"].position[0] > by_color["blue"].position[0]
    assert {trk.id for trk in tr.confirmed} == {1, 2}


def test_same_colour_crossing_keeps_ids():
    """Two red balls cross; while they overlap the detector sees ONE blob in the middle.
    The constant-velocity prediction carries each track through the crossing."""
    tr = MultiTracker(process_noise=300.0)
    t = 0.0
    ids_at_start = None
    for k in range(50):
        a = np.array([100 + 9 * k, 200 + 0.5 * k])     # moves right
        b = np.array([550 - 9 * k, 215 - 0.5 * k])     # moves left
        if np.linalg.norm(a - b) < 30:
            dets = [Detection(*((a + b) / 2), 22, "red")]
        else:
            dets = [Detection(*a, 15, "red"), Detection(*b, 15, "red")]
        tr.step(dets, t)
        t += DT
        if k == 10:
            ids_at_start = {trk.id: trk.velocity[0] > 0 for trk in tr.confirmed}
    ids_at_end = {trk.id: trk.velocity[0] > 0 for trk in tr.confirmed}
    assert ids_at_end == ids_at_start     # the right-mover keeps its id, and so does the left-mover


def test_pendulum_tracking_accuracy():
    """Pendulum swing (amplitude ~170 px, period 1.6 s, up to ~2800 px/s^2), 1.5 px noise and
    10 % missed frames: a single id the whole time and an error of a few px.
    Regression: with only the strict gate the track was dropped at the ends of the swing
    (31 ids in 8 s); the second-chance association keeps it."""
    rng = np.random.default_rng(2)
    tr = MultiTracker(process_noise=3000.0, meas_noise=1.5)
    t, err_f, err_m, ids = 0.0, [], [], set()
    w = 2 * np.pi / 1.6
    for k in range(240):
        th = 0.6 * np.sin(w * t)
        truth = np.array([320 + 300 * np.sin(th), 60 + 300 * np.cos(th)])
        meas = truth + rng.normal(0, 1.5, 2)
        dets = [] if rng.random() < 0.1 else [Detection(*meas, 18, "red")]
        tr.step(dets, t)
        if k > 30 and tr.confirmed:
            ids |= {trk.id for trk in tr.confirmed}
            err_f.append(np.linalg.norm(tr.confirmed[0].position - truth))
            err_m.append(np.linalg.norm(meas - truth))
        t += DT
    assert ids == {1}
    assert np.sqrt(np.mean(np.square(err_f))) < 3.5


def test_bbox_follows_radius():
    tr = MultiTracker()
    run(tr, [[Detection(100, 100, 20, "red")]] * 3)
    x0, y0, x1, y1 = tr.confirmed[0].bbox()
    assert (x1 - x0) == pytest.approx(40, abs=2) and (y1 - y0) == pytest.approx(40, abs=2)
