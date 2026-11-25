"""Decoder tests with simulated per-frame label sequences (no camera)."""
import numpy as np
import pytest

from decoder import (ARMED, ENTERING, LOCKED, PENALTY, SequenceDecoder, select_label)

CODE = ["TRIANGLE", "CIRCLE", "DOTS", "DIAMOND"]
FPS = 30.0
DT = 1 / FPS


def make():
    return SequenceDecoder(CODE, window_frames=10, agree_frac=0.6, timeout_s=8.0,
                           max_failures=3, penalties_s=[10, 20, 40])


def feed(dec, label, seconds, t):
    """Feed the same label for some seconds at 30 FPS. Returns the new time."""
    for _ in range(int(round(seconds * FPS))):
        dec.update(label, t)
        t += DT
    return t


def show(dec, label, t, hold=0.6, gap=0.5):
    """Operator shows a card for `hold` seconds and then removes it for `gap` seconds."""
    t = feed(dec, label, hold, t)
    return feed(dec, None, gap, t)


def enter(dec, labels, t):
    for lab in labels:
        t = show(dec, lab, t)
    return t


def test_correct_code_arms():
    dec = make()
    t = enter(dec, CODE, 0.0)
    assert dec.state == ARMED and dec.armed
    assert [e.name for e in dec.events] == ["read"] * 4 + ["armed"]
    assert t < 5.0  # 4 cards at ~1.1 s each


def test_wrong_code_is_only_rejected_after_four_cards():
    dec = make()
    t = enter(dec, ["CIRCLE", "CIRCLE", "DOTS"], 0.0)   # first card already wrong
    assert dec.state == ENTERING and len(dec.entered) == 3   # no early rejection
    enter(dec, ["DIAMOND"], t)
    assert dec.state == LOCKED and dec.failures == 1 and dec.entered == []


def test_holding_a_card_counts_once():
    dec = make()
    feed(dec, "TRIANGLE", 5.0, 0.0)
    assert dec.entered == ["TRIANGLE"]


def test_same_symbol_twice_needs_removal_in_between():
    dec = SequenceDecoder(["DOTS", "DOTS"], window_frames=10)
    t = feed(dec, "DOTS", 1.0, 0.0)
    t = feed(dec, None, 0.15, t)          # too short to count as removed
    feed(dec, "DOTS", 1.0, t)
    assert dec.entered == ["DOTS"]
    dec2 = SequenceDecoder(["DOTS", "DOTS"], window_frames=10)
    enter(dec2, ["DOTS", "DOTS"], 0.0)
    assert dec2.state == ARMED


def test_brief_glimpse_is_not_read():
    dec = make()
    t = feed(dec, "TRIANGLE", 0.15, 0.0)  # ~4 frames
    feed(dec, None, 1.0, t)
    assert dec.entered == [] and dec.state == LOCKED


def test_flickering_detection_is_still_read():
    """The detector misses 1 frame in 6: the card is still read (once)."""
    dec = make()
    t = 0.0
    for lab in CODE:
        for k in range(24):
            dec.update(None if k % 6 == 5 else lab, t)
            t += DT
        t = feed(dec, None, 0.5, t)
    assert dec.state == ARMED


def test_noisy_detector_unlocks_and_never_false_accepts():
    """Random drop-outs (30 %) and UNKNOWN frames (10 %). The operator holds each card until
    it is read (the UI shows a progress bar), at most 3 s, and then removes it.
    The right code always arms, a wrong one never does, and each card is read exactly once."""
    rng = np.random.default_rng(0)
    times = []
    for code_ok in (True, False):
        for _ in range(50):
            dec = make()
            seq = CODE if code_ok else ["TRIANGLE", "CIRCLE", "DOTS", "PENTAGON"]
            t = 0.0
            for k, lab in enumerate(seq):
                for _ in range(int(3 * FPS)):
                    r = rng.random()
                    dec.update(None if r < 0.3 else ("UNKNOWN" if r < 0.4 else lab), t)
                    t += DT
                    if len(dec.entered) > k or dec.state != ENTERING and k == len(seq) - 1:
                        break
                t = feed(dec, None, 0.6, t)
            assert dec.armed == code_ok
            assert sum(e.name == "read" for e in dec.events) == 4
            if code_ok:
                times.append(t)
    assert np.mean(times) < 5.0   # 4 cards in under 5 s on average even with a bad detector


def test_cancel_clears_the_sequence():
    dec = make()
    t = enter(dec, ["TRIANGLE", "CIRCLE"], 0.0)
    t = show(dec, "CANCEL", t)
    assert dec.state == LOCKED and dec.entered == []
    enter(dec, CODE, t)
    assert dec.armed and dec.failures == 0   # cancelling is not a failed attempt


def test_timeout_clears_partial_sequence():
    dec = make()
    t = enter(dec, ["TRIANGLE", "CIRCLE"], 0.0)
    t = feed(dec, None, 9.0, t)
    assert dec.state == LOCKED and dec.entered == []
    assert any(e.name == "timeout" for e in dec.events)
    enter(dec, CODE, t)
    assert dec.armed


def test_three_failures_trigger_escalating_penalties():
    dec = make()
    wrong = ["CIRCLE"] * 4
    t = 0.0
    for _ in range(3):
        t = enter(dec, wrong, t)
    assert dec.state == PENALTY
    assert dec.remaining(t) == pytest.approx(10, abs=1.0)
    # During the penalty even the right code is ignored
    t = enter(dec, CODE, t)
    assert dec.state == PENALTY and not dec.armed
    t = feed(dec, None, 10.0, t)
    assert dec.state == LOCKED
    for _ in range(3):
        t = enter(dec, wrong, t)
    assert dec.state == PENALTY and dec.remaining(t) == pytest.approx(20, abs=1.0)


def test_reset_does_not_shorten_penalty():
    dec = make()
    t = 0.0
    for _ in range(3):
        t = enter(dec, ["CIRCLE"] * 4, t)
    dec.reset(t)
    assert dec.state == PENALTY


def test_reset_clears_and_disarms():
    dec = make()
    t = enter(dec, ["TRIANGLE"], 0.0)
    dec.reset(t)
    assert dec.state == LOCKED and dec.entered == []
    t = enter(dec, CODE, t)
    assert dec.armed
    dec.reset(t)
    assert dec.state == LOCKED and not dec.armed


def test_success_resets_failure_counter():
    dec = make()
    t = enter(dec, ["CIRCLE"] * 4, 0.0)
    t = enter(dec, ["CIRCLE"] * 4, t)
    t = enter(dec, CODE, t)
    assert dec.armed and dec.failures == 0


def test_select_label():
    assert select_label([]) is None
    assert select_label([("CIRCLE", 900.0)]) == "CIRCLE"
    assert select_label([("CIRCLE", 900.0), ("UNKNOWN", 400.0)]) == "CIRCLE"
    assert select_label([("CIRCLE", 900.0), ("DOTS", 400.0)]) is None   # two cards at once: ambiguous
    assert select_label([("UNKNOWN", 900.0)]) is None


@pytest.mark.parametrize("code", [[], ["CIRCLE"] * 5, ["CANCEL"], ["SQUARE"]])
def test_invalid_codes_are_rejected(code):
    with pytest.raises(ValueError):
        SequenceDecoder(code)


def test_from_config():
    cfg = {"decoder": {"code": ["dots", "bars"], "window_frames": 6}}
    dec = SequenceDecoder.from_config(cfg)
    assert dec.code == ["DOTS", "BARS"] and dec.window.maxlen == 6
