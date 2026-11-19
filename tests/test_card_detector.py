"""Card detector tests with ideal cards and synthetic scenes."""
import numpy as np
import pytest

from card_detector import CardDetector, order_corners
from card_symbols import CANCEL, DIAMOND, SYMBOLS, UNKNOWN, render_card
from synth_cards import background, card_scene

DET = CardDetector()


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_ideal_rectified_card(symbol):
    """An already rectified card (no camera) is classified correctly: validates the rules."""
    rect = render_card(symbol, 200, margin=False)
    f = DET.symbol_features(DET.symbol_mask(rect))
    assert DET.classify(f) == symbol


@pytest.mark.parametrize("symbol", SYMBOLS)
@pytest.mark.parametrize("seed", range(6))
def test_card_in_scene(symbol, seed):
    """Perspective (up to 40°), free rotation, uneven lighting, blur, noise and distractors."""
    rng = np.random.default_rng(1000 * seed + SYMBOLS.index(symbol))
    img, frame = card_scene(symbol, rng)
    dets = DET.detect(img)
    assert len(dets) == 1, f"{len(dets)} detections"
    assert dets[0].label == symbol, dets[0].features
    # Detected corners match the true ones (within a few pixels). The starting corner
    # may vary if the card is rotated ~45°: every cyclic shift is tried
    true = order_corners(frame)
    err = min(np.linalg.norm(np.roll(true, k, axis=0) - dets[0].quad, axis=1).max() for k in range(4))
    assert err < 4.0


def test_no_card_no_detection():
    rng = np.random.default_rng(7)
    for _ in range(20):
        assert DET.detect(background(rng)) == []


def test_empty_frame_is_not_a_card():
    """A frame without a symbol (e.g. a picture on the wall) does not count as a card."""
    img = np.full((480, 640), 150, np.uint8)
    img[150:330, 230:410] = render_card(None, 180)
    assert DET.detect(img) == []


def test_unknown_symbol_is_reported_as_unknown():
    """A frame with a symbol that is not part of the design is detected as UNKNOWN."""
    img = render_card(None, 180)
    img[70:110, 50:130] = 0          # thick horizontal rectangle: matches no rule
    scene = np.full((480, 640), 150, np.uint8)
    scene[150:330, 230:410] = img
    dets = DET.detect(scene)
    assert len(dets) == 1 and dets[0].label == UNKNOWN


def test_card_outside_roi_is_ignored():
    img = np.full((480, 640), 150, np.uint8)
    img[5:125, 5:125] = render_card(CANCEL, 120)   # top-left corner, outside the ROI
    assert DET.detect(img) == []


def test_order_corners_clockwise_from_top_left():
    pts = np.array([[10, 90], [90, 90], [10, 10], [90, 10]], float)
    np.testing.assert_allclose(order_corners(pts), [[10, 10], [90, 10], [90, 90], [10, 90]])


def test_debug_stages_are_saved():
    img = np.full((480, 640), 150, np.uint8)
    img[150:330, 230:410] = render_card(SYMBOLS[0], 180)
    dbg = {}
    DET.detect(img, dbg)
    assert {"1_gray", "2_adaptive_threshold", "3_contours", "4_rectified", "5_symbol"} <= set(dbg)


def test_no_confusions_in_hard_conditions():
    """Small card, tilted up to 55° and blurred: it may not be read, but it must NEVER
    be read as another symbol (the decoder must not receive a wrong class)."""
    wrong, read = [], 0
    for symbol in SYMBOLS:
        rng = np.random.default_rng(500 + SYMBOLS.index(symbol))
        for _ in range(25):
            img, _ = card_scene(symbol, rng, max_tilt_deg=55, side_px=(60, 140), blur=(0.5, 2.0))
            for d in DET.detect(img):
                if d.label == symbol:
                    read += 1
                elif d.label != UNKNOWN:
                    wrong.append((symbol, d.label, d.features))
    assert wrong == []
    assert read >= 0.9 * 25 * len(SYMBOLS)


def test_diamond_survives_blur():
    """Regression: with blur the corners get rounded and the diamond fill rises to ~0.79;
    it used to fall in the pentagon range and come out as UNKNOWN."""
    rng = np.random.default_rng(5)
    ok = 0
    for _ in range(40):
        img, _ = card_scene(DIAMOND, rng, max_tilt_deg=55, side_px=(60, 140), blur=(1.0, 2.0))
        dets = DET.detect(img)
        ok += len(dets) == 1 and dets[0].label == DIAMOND
    assert ok >= 36
