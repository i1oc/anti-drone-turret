# Virtual anti-drone turret with real targets

Final project for **Computer Vision I** · 3rd year, Mathematical Engineering · Universidad Pontificia Comillas ICAI · academic year 2026/27.

A fixed real camera watches an area where physical targets (coloured balls hanging like pendulums) move around acting as "drones". The system stays **locked** until the operator shows the camera a sequence of 4 printed cards in the right order. Once **armed**, it detects and tracks each target with a Kalman filter, and a **virtual** turret drawn on screen rotates, aims with lead and fires a virtual projectile when it predicts a hit.

Everything is classical computer vision with Python and OpenCV, with no deep learning and no extra hardware.

> **Status:** real-time camera with FPS, offline calibration with distortion correction and card detector.

## Team

| Member | Main module |
| --- | --- |
| _First Last_ | Calibration and geometry |
| _First Last_ | Cards and decoder |
| _First Last_ | Detection and tracking |
| _First Last_ | Integration, evaluation and documentation |

Group: _A / B_

## Installation

Requirements: **Python 3.9** and Git. Library versions are those of the course lab environment ([icai-voi/voi-26-27](https://github.com/icai-voi/voi-26-27)); OpenCV 4.8 and NumPy 1.26 have no builds for Python 3.13 or later.

```bash
git clone https://github.com/<user>/torreta-antidron.git
cd torreta-antidron

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
| `c` | Toggle the card detector |
| `d` | Save the card detector stages to `data/debug/` |

### Arming cards

```bash
python cards/make_cards.py   # PDF with the 8 cards of 90 mm (print at 100 %)
```

Each card is a thick black frame with a symbol: circle, bullseye, triangle, diamond, pentagon,
3 bars, 4 dots and an X (CANCEL). The detector looks for the frame (adaptive threshold + contours +
filter for quadrilaterals with a hole), straightens it with a homography and classifies the symbol with
rotation-invariant features (number of pieces and holes, fill relative to the enclosing circle, solidity and vertices).

## Repository structure

```text
torreta-antidron/
├── src/                # system code
│   ├── main.py         # entry point and real-time loop
│   ├── camera.py       # capture in a separate thread
│   ├── camera_model.py # distortion correction, pixel -> angles, depth
│   ├── calib_utils.py  # board detection, calibration, validation RMS
│   ├── card_symbols.py # card design (single definition)
│   ├── card_detector.py# card detector and classifier
│   ├── settings.py     # paths and config loading
│   └── fps.py          # FPS and performance overlay
├── config/config.yaml  # parameters (camera, thresholds, arming code...)
├── calibration/        # calibration scripts, board images and calib.yaml
├── cards/              # card PDF generator
├── tests/              # pytest tests
├── docs/figures/       # report figures
└── data/               # test videos and captures (not committed)
```

## Roadmap

- [x] Real-time camera capture with FPS
- [x] Offline calibration (K, distortion, RMS) and distortion correction (RMS 0.22 px, validation 0.23 px)
- [x] Arming card detector
- [ ] Sequence decoder (state machine)
- [ ] Multi-target detection and tracking (Kalman + Hungarian assignment)
- [ ] Fire control: virtual turret, lead aiming, IFF, radar
- [ ] Quantitative evaluation and demo video

## Use of AI assistants

Claude (Anthropic) was used as a design and programming assistant. All code has been reviewed, tested and understood by the team.
