"""Generate the calibration chessboard as a true-scale A4 PDF.

Uses the measurements from config.yaml (calibration.squares_x, squares_y, square_mm).
Includes a 50 mm ruler to check with a real ruler that the printer has not
scaled the page (print at 100 %, no "fit to page").

Usage:
    python calibration/make_chessboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from settings import ROOT, load_config  # noqa: E402

MM_PER_INCH = 25.4


def main() -> None:
    c = load_config()["calibration"]
    sx, sy, sq = c["squares_x"], c["squares_y"], float(c["square_mm"])
    page_w, page_h = 297.0, 210.0  # A4 landscape, in mm
    board_w, board_h = sx * sq, sy * sq
    if board_w > page_w - 20 or board_h > page_h - 30:
        raise SystemExit(f"The board ({board_w}x{board_h} mm) does not fit on an A4 page with margins")

    fig = plt.figure(figsize=(page_w / MM_PER_INCH, page_h / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)  # origin at the top left, in mm
    ax.axis("off")

    x0 = (page_w - board_w) / 2
    y0 = (page_h - board_h) / 2 - 6
    for r in range(sy):
        for col in range(sx):
            if (r + col) % 2 == 0:
                ax.add_patch(Rectangle((x0 + col * sq, y0 + r * sq), sq, sq, color="black", lw=0))

    # 50 mm ruler with ticks every 10 mm
    ry = y0 + board_h + 9
    ax.plot([x0, x0 + 50], [ry, ry], color="black", lw=1)
    for k in range(6):
        ax.plot([x0 + 10 * k] * 2, [ry - 2.5, ry], color="black", lw=1)
    ax.text(x0 + 55, ry, "50 mm: check with a ruler", va="center", fontsize=8)
    ax.text(x0 + board_w, ry, f"{sx}x{sy} squares of {sq:g} mm  ·  print at 100 %",
            va="center", ha="right", fontsize=8)

    out = ROOT / "calibration" / f"chessboard_{sx}x{sy}_{sq:g}mm.pdf"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
