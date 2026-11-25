# Virtual anti-drone turret with real targets

Final project for **Computer Vision I** · 3rd year, Mathematical Engineering · Universidad Pontificia Comillas ICAI · academic year 2026/27.

A fixed real camera watches an area where physical targets (coloured balls hanging like pendulums) move around acting as "drones". The system stays **locked** until the operator shows the camera a sequence of 4 printed cards in the right order. Once **armed**, it detects and tracks each target with a Kalman filter, and a **virtual** turret drawn on screen rotates, aims with lead and fires a virtual projectile when it predicts a hit.

Everything is classical computer vision with Python and OpenCV, with no deep learning and no extra hardware.

> **Status:** real-time camera with FPS, offline calibration with distortion correction, card detector, sequence decoder and multi-target tracker.

## Team

- Hugo Lara Rodríguez
- Pablo Santiago Hernández Muñoz

Group: A

## Installation

Requirements: **Python 3.9** and Git. Library versions are those of the course lab environment ([icai-voi/voi-26-27](https://github.com/icai-voi/voi-26-27)); OpenCV 4.8 and NumPy 1.26 have no builds for Python 3.13 or later.

```bash
git clone https://github.com/i1oc/anti-drone-turret.git
cd anti-drone-turret

# Windows
py -3.9 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.9 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py            # camera set in config/config.yaml
python src/main.py --camera 1 # another camera
pytest                        # tests
```

### Camera calibration (offline, once)

Board of 11×8 squares of 20 mm (10×7 inner corners); the measurements are in `config/config.yaml`.

```bash
python calibration/make_chessboard.py   # A4 PDF: print at 100 % and check the 50 mm ruler
python calibration/capture.py           # SPACE saves, z undoes, q quits (20-30 images)
python calibration/calibrate.py         # K, distortion, RMS and validation RMS -> calibration/calib.yaml
```

`calibrate.py` holds out 20 % of the images to measure the **validation RMS** (reprojection error on
unused images), compares the 4- and 5-coefficient distortion models and saves the report figures to
`docs/figures/calibration/`. `--no-curve` skips the error vs number of images curve, which is the slow part.

| Key | Action |
| --- | --- |
| `q` / `Esc` | Quit |
| `s` | Save the current frame to `data/snapshots/` |
| `u` | Toggle distortion correction (requires `calibration/calib.yaml`) |
| `c` | Toggle the card overlay (frames, labels, features) |
| `d` | Save the card detector stages to `data/debug/` |
| `r` | Reset the sequence / disarm (does not shorten a penalty) |

### Arming cards

```bash
python cards/make_cards.py   # PDF with the 8 cards of 90 mm (print at 100 %)
```

Each card is a thick black frame with a symbol: circle, bullseye, triangle, diamond, pentagon,
3 bars, 4 dots and an X (CANCEL). The detector looks for the frame (adaptive threshold + contours +
filter for quadrilaterals with a hole), straightens it with a homography and classifies the symbol with
rotation-invariant features (number of pieces and holes, fill relative to the enclosing circle, solidity and vertices).

### Sequence decoder

The system starts **LOCKED**. Show the 4 cards of the code (`decoder.code` in `config/config.yaml`) one by one,
holding each until the progress bar fills and removing it before the next one:

```text
LOCKED -> ENTERING (1..3 cards) -> VERIFY (4 cards) -> ARMED
                                        |  wrong code (x3)
                                        v
                                     PENALTY (10 / 20 / 40 s) -> LOCKED
```

- A card is read when it appears in 6 of the last 10 frames, and it must be removed (10 empty frames)
  before the next read (hysteresis: holding a card counts once).
- The code is checked only after the 4th card, so a wrong first card is not revealed.
- `CANCEL` card or `r` clears the sequence; 8 s without a new card clears it too.
- Events are saved to `data/logs/decoder_*.csv` on exit (unlock time experiment).

### Ball tracker (when ARMED)

```bash
python tools/pick_colors.py   # tune the colours: select with 1-2, click on the ball, 'w' saves config/colors.yaml
```

Targets: a mini fluorescent **tennis** ball (friend, never engaged) and a mini orange **basketball** (hostile).

- **Detection:** HSV colour ranges (`cv2.inRange`) -> opening/closing -> contours ->
  circularity >= 0.82 and fill >= 0.65 (rejects squares and other shapes) -> `minEnclosingCircle`.
- **Basketball seams:** a 9 px closing fills the black seams inside the ball and the shape is measured on
  the convex hull (seams reaching the edge leave notches; a rectangle stays a rectangle).
- Optional per-colour **motion confirmation with MOG2** (for a colour the background also has); off for
  both balls, so MOG2 does not run.
- **Tracking:** one Kalman filter per target (constant velocity; constant acceleration selectable),
  squared Mahalanobis distance with a chi-square 95 % gate, Hungarian assignment
  (`scipy.optimize.linear_sum_assignment`) per colour, plus a second chance with a wider gate for
  established tracks (a pendulum accelerates hard at the ends of the swing).
- **Life cycle:** tentative -> confirmed after 3 hits (gets a visible id) -> lost (coasting on the
  prediction) -> deleted after 10 misses.
- **On screen:** bounding box, id, colour (green = FRIEND), trail, velocity arrow (0.25 s ahead), speed,
  azimuth/elevation from the calibration and depth `Z = f*D/d` (`diameter_m` of each colour: measure your balls).

## Repository structure

```text
anti-drone-turret/
├── src/                # system code
│   ├── main.py         # entry point and real-time loop
│   ├── camera.py       # capture in a separate thread
│   ├── camera_model.py # distortion correction, pixel -> angles, depth
│   ├── calib_utils.py  # board detection, calibration, validation RMS
│   ├── card_symbols.py # card design (single definition)
│   ├── card_detector.py# card detector and classifier
│   ├── decoder.py      # sequence decoder (state machine, no OpenCV)
│   ├── ball_detector.py# coloured ball detection (HSV)
│   ├── tracking/       # Kalman, association, track management (no OpenCV)
│   ├── hud.py          # on-screen panels
│   ├── settings.py     # paths and config loading
│   └── fps.py          # FPS and performance overlay
├── config/config.yaml  # parameters (camera, thresholds, arming code...)
├── calibration/        # calibration scripts, board images and calib.yaml
├── cards/              # card PDF generator
├── tools/              # pick_colors.py (HSV tuning by clicking)
├── tests/              # pytest tests
├── docs/figures/       # report figures
└── data/               # test videos and captures (not committed)
```

## Roadmap

- [x] Real-time camera capture with FPS
- [x] Offline calibration (K, distortion, RMS) and distortion correction (RMS 0.22 px, validation 0.23 px)
- [x] Arming card detector
- [x] Sequence decoder (state machine)
- [x] Multi-target detection and tracking (Kalman + Hungarian assignment)
- [ ] Fire control: virtual turret, lead aiming, IFF, radar
- [ ] Quantitative evaluation and demo video

## Use of AI assistants

Claude (Anthropic) was used as a design and programming assistant. All code has been reviewed, tested and understood by the team.
