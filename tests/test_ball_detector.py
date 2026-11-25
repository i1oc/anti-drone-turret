"""Ball detector tests with synthetic images: mini fluorescent tennis ball and mini orange
basketball (with black seams), plus distractors. The generic per-colour options
(motion confirmation with MOG2, convex hull) are tested with a local white-ball colour."""
import cv2
import numpy as np
import pytest

from ball_detector import BallDetector, hsv_ranges_from_samples
from settings import load_config

CFG = load_config()
COLORS = CFG["tracker"]["colors"]
BGR = {"tennis": (40, 230, 200), "basketball": (20, 120, 255)}


def detector(colors=COLORS):
    return BallDetector(colors, min_area=80, min_circularity=0.82, min_fill=0.65)


def scene(rng=None, bg=(110, 105, 100)):
    rng = rng or np.random.default_rng(0)
    img = np.zeros((480, 640, 3), np.float32)
    img[:] = bg
    img += rng.normal(0, 4, img.shape)
    return img


def draw_ball(img, x, y, r, bgr):
    """Shaded ball: darker towards the edge, with a white specular highlight."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
    inside = d <= 1
    shade = (1.0 - 0.35 * d ** 2)[..., None]
    img[inside] = (np.array(bgr, np.float32) * shade)[inside]
    cv2.circle(img, (int(x - 0.35 * r), int(y - 0.35 * r)), max(2, int(0.18 * r)), (250, 250, 250), -1)


def draw_basketball(img, x, y, r, rot=0.0):
    """Orange ball with black seams: two great circles through the ball and two curved lines,
    all of them reaching the edge (they cut notches into the orange mask)."""
    draw_ball(img, x, y, r, BGR["basketball"])
    t = max(1, int(round(0.08 * r)))
    c, s = np.cos(rot), np.sin(rot)
    for ang in (0, np.pi / 2):                                   # straight seams
        dx, dy = r * np.cos(ang + rot), r * np.sin(ang + rot)
        cv2.line(img, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), (20, 20, 20), t, cv2.LINE_AA)
    for side in (-1, 1):                                         # curved side seams
        cx, cy = x + side * 1.3 * r * c, y + side * 1.3 * r * s
        cv2.circle(img, (int(cx), int(cy)), int(0.95 * r), (20, 20, 20), t, cv2.LINE_AA)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]          # keep only what lies on the ball
    return (xx - x) ** 2 + (yy - y) ** 2 <= r * r


def basketball_scene(x, y, r, rot=0.0, rng=None):
    img = scene(rng)
    bg = img.copy()
    inside = draw_basketball(img, x, y, r, rot)
    img[~inside] = bg[~inside]
    return img


def finish(img):
    return np.clip(cv2.GaussianBlur(img, (3, 3), 0), 0, 255).astype(np.uint8)


def test_tennis_ball_is_detected():
    img = scene()
    draw_ball(img, 300, 220, 25, BGR["tennis"])
    dets = detector().detect(finish(img))
    assert len(dets) == 1 and dets[0].color == "tennis"
    assert abs(dets[0].x - 300) < 2 and abs(dets[0].y - 220) < 2 and abs(dets[0].r - 25) < 3


@pytest.mark.parametrize("rot", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("r", [12, 25, 40])
def test_basketball_with_seams_is_detected(r, rot):
    """Mini basketball at several sizes and rotations: the seams do not break the detection."""
    dets = detector().detect(finish(basketball_scene(320, 240, r, rot)))
    assert len(dets) == 1 and dets[0].color == "basketball", dets
    assert abs(dets[0].x - 320) < 2 and abs(dets[0].y - 240) < 2 and abs(dets[0].r - r) < 0.15 * r + 2


def test_basketball_needs_the_convex_hull():
    """Without use_hull the seams on the edge cut the blob (lower circularity)."""
    img = finish(basketball_scene(320, 240, 30, 0.3))
    plain = {"basketball": {"ranges": COLORS["basketball"]["ranges"], "close_px": 9}}
    hull = {"basketball": {"ranges": COLORS["basketball"]["ranges"], "close_px": 9, "use_hull": True}}
    strict = BallDetector(plain, min_circularity=0.9, min_fill=0.8)
    assert strict.detect(img) == [] and len(BallDetector(hull, min_circularity=0.9, min_fill=0.8).detect(img)) == 1


def test_hues_do_not_mix():
    """The tennis ball is not a basketball and vice versa."""
    for color in BGR:
        img = scene()
        draw_ball(img, 320, 240, 25, BGR[color])
        assert [d.color for d in detector().detect(finish(img))] == [color]


def test_distractors_are_rejected():
    img = basketball_scene(150, 150, 22)
    draw_ball(img, 450, 300, 30, BGR["tennis"])
    cv2.rectangle(img, (500, 40), (560, 100), BGR["basketball"], -1)   # orange square: hull stays a square
    cv2.rectangle(img, (40, 380), (200, 420), BGR["tennis"], -1)       # tennis-coloured bar
    cv2.circle(img, (560, 420), 25, (120, 150, 200), -1)               # skin-like: saturation too low
    cv2.circle(img, (80, 60), 3, BGR["tennis"], -1)                    # tiny speck
    dets = detector().detect(finish(img))
    assert sorted(d.color for d in dets) == ["basketball", "tennis"]


def test_no_motion_model_when_no_colour_needs_it():
    assert detector().mog2 is None   # saves the MOG2 cost every frame


def test_motion_confirm_option():
    """Generic option (for a colour that the background also has, e.g. a white ball):
    the moving white ball is detected, a static white disc is not."""
    white = {"white": {"ranges": [[[0, 0, 160], [180, 55, 255]]], "motion_confirm": True}}
    det = detector(white)
    bg = scene()
    cv2.circle(bg, (520, 90), 32, (240, 240, 240), -1)                 # static white disc
    for k in range(40):
        img = bg.copy()
        draw_ball(img, 120 + 8 * k, 300, 30, (235, 235, 235))
        dets = det.detect(finish(img))
        if k >= 20:
            assert [(d.color, abs(d.x - (120 + 8 * k)) < 5) for d in dets] == [("white", True)]


def test_debug_stages():
    img = scene()
    draw_ball(img, 300, 220, 25, BGR["basketball"])
    dbg = {}
    detector().detect(finish(img), dbg)
    assert {"1_hsv_hue", "2_mask_basketball", "3_mask_clean_tennis", "4_contours"} <= set(dbg)


def test_ranges_from_samples():
    rng = np.random.default_rng(0)
    tennis = np.column_stack([rng.normal(35, 3, 500), rng.normal(180, 20, 500), rng.normal(200, 25, 500)])
    (lo, hi), = hsv_ranges_from_samples(np.clip(tennis, 0, 255))
    assert lo[0] < 30 and hi[0] > 40 and hi[0] - lo[0] <= 44
    red = np.column_stack([np.concatenate([rng.uniform(0, 4, 250), rng.uniform(176, 180, 250)]),
                           np.full(500, 200), np.full(500, 160)])
    ranges = hsv_ranges_from_samples(red)
    assert len(ranges) == 2                                            # split across 0/180
    covered = lambda h: any(r[0][0] <= h <= r[1][0] for r in ranges)  # noqa: E731
    assert covered(1) and covered(178) and not covered(90)
    white = np.column_stack([rng.uniform(0, 180, 500), rng.uniform(0, 30, 500), rng.uniform(190, 250, 500)])
    (lo, hi), = hsv_ranges_from_samples(white)                         # achromatic: any hue, low S, high V
    assert lo[0] == 0 and hi[0] == 180 and hi[1] <= 90 and lo[2] >= 100


def draw_real_tennis(img, x, y, r):
    """Tennis ball as seen by the real camera: pale, overexposed top (S ~50, V 255) and a
    white seam crossing the whole ball."""
    draw_ball(img, x, y, r, (60, 225, 205))
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    top = inside & (yy < y - 0.2 * r)
    img[top] = 0.35 * img[top] + 0.65 * np.array([200, 255, 255], np.float32)    # burnt highlight
    cv2.ellipse(img, (int(x + 0.6 * r), int(y - 0.5 * r)), (int(0.9 * r), int(0.7 * r)), 20, 100, 260,
                (225, 235, 230), max(2, int(0.16 * r)), cv2.LINE_AA)            # white seam
    bg = scene()
    img[~inside] = bg[~inside]


@pytest.mark.parametrize("r", [15, 30, 60])
def test_real_tennis_ball_overexposed_with_seam(r):
    """Regression with the real ball: the default range (S >= 90) lost the pale top, and the
    seam broke the blob. Now: S >= 40, 15 px closing and convex hull."""
    img = scene()
    draw_real_tennis(img, 320, 240, r)
    dets = detector().detect(finish(img))
    assert [d.color for d in dets] == ["tennis"]
    assert abs(dets[0].r - r) < 0.15 * r + 2


@pytest.mark.parametrize("r", [40, 90])
def test_basketball_split_by_a_wide_seam_is_grouped(r):
    """Close to the camera a black seam is wide enough to cut the ball in two halves:
    the fragments are grouped (group_px) and the hull of the group is one round ball."""
    img = scene()
    bg = img.copy()
    inside = draw_basketball(img, 320, 240, r, 0.0)
    cv2.ellipse(img, (320, 240 + int(0.9 * r)), (int(1.1 * r), int(0.75 * r)), 0, 180, 360,
                (20, 20, 20), max(3, int(0.22 * r)), cv2.LINE_AA)                # wide seam across the ball
    img[~inside] = bg[~inside]
    dets = detector().detect(finish(img))
    assert [d.color for d in dets] == ["basketball"]
    assert abs(dets[0].r - r) < 0.15 * r + 2 and abs(dets[0].y - 240) < 0.15 * r
