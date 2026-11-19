"""Generate the 8 arming cards as a PDF (A4, 4 cards per page, true scale).

The design comes from src/card_symbols.py (the same definition the detector
uses), so what is printed matches what is expected. The card size is taken
from config.yaml (cards.card_mm, white margin included).

Print at 100 % (no "fit to page"), cut along the grey line and, if possible,
glue onto cardboard so they do not bend.

Usage:
    python cards/make_cards.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from card_symbols import SYMBOLS, render_card  # noqa: E402
from settings import ROOT, load_config  # noqa: E402

MM_PER_INCH = 25.4
PAGE_W, PAGE_H = 210.0, 297.0  # A4 portrait, in mm


def main() -> None:
    card = float(load_config()["cards"]["card_mm"])
    gap = 12.0
    if 2 * card + gap > PAGE_W - 10 or 2 * card + 2 * gap > PAGE_H - 20:
        raise SystemExit(f"{card} mm cards: 4 do not fit on an A4 page")
    x0 = (PAGE_W - (2 * card + gap)) / 2
    y0 = (PAGE_H - (2 * card + 2 * gap)) / 2

    out = ROOT / "cards" / f"cards_{card:g}mm.pdf"
    with PdfPages(out) as pdf:
        for page in range(0, len(SYMBOLS), 4):
            fig = plt.figure(figsize=(PAGE_W / MM_PER_INCH, PAGE_H / MM_PER_INCH))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, PAGE_W)
            ax.set_ylim(PAGE_H, 0)
            ax.axis("off")
            for k, name in enumerate(SYMBOLS[page:page + 4]):
                cx = x0 + (k % 2) * (card + gap)
                cy = y0 + (k // 2) * (card + gap)
                img = render_card(name, size_px=1200)  # ~340 dpi for 90 mm
                ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                          extent=(cx, cx + card, cy + card, cy))
                ax.add_patch(plt.Rectangle((cx, cy), card, card, fill=False, ec="0.75", lw=0.4))  # cut line
                ax.text(cx + card / 2, cy + card + 4, name, ha="center", va="center", fontsize=9, color="0.4")
            ax.text(PAGE_W / 2, PAGE_H - 8, f"{card:g} mm cards · print at 100 %",
                    ha="center", fontsize=8, color="0.4")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
