"""Genera el tablero de calibración en PDF A4 a escala real.

Usa las medidas de config.yaml (calibration.squares_x, squares_y, square_mm).
Incluye una regla de 50 mm para comprobar con una regla de verdad que la
impresora no ha escalado la hoja (imprimir al 100 %, sin "ajustar a página").

Uso:
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
    page_w, page_h = 297.0, 210.0  # A4 apaisado, en mm
    board_w, board_h = sx * sq, sy * sq
    if board_w > page_w - 20 or board_h > page_h - 30:
        raise SystemExit(f"El tablero ({board_w}x{board_h} mm) no cabe en un A4 con márgenes")

    fig = plt.figure(figsize=(page_w / MM_PER_INCH, page_h / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)  # origen arriba a la izquierda, en mm
    ax.axis("off")

    x0 = (page_w - board_w) / 2
    y0 = (page_h - board_h) / 2 - 6
    for r in range(sy):
        for col in range(sx):
            if (r + col) % 2 == 0:
                ax.add_patch(Rectangle((x0 + col * sq, y0 + r * sq), sq, sq, color="black", lw=0))

    # Regla de 50 mm con marcas cada 10 mm
    ry = y0 + board_h + 9
    ax.plot([x0, x0 + 50], [ry, ry], color="black", lw=1)
    for k in range(6):
        ax.plot([x0 + 10 * k] * 2, [ry - 2.5, ry], color="black", lw=1)
    ax.text(x0 + 55, ry, "50 mm: comprobar con una regla", va="center", fontsize=8)
    ax.text(x0 + board_w, ry, f"{sx}x{sy} cuadros de {sq:g} mm  ·  imprimir al 100 %",
            va="center", ha="right", fontsize=8)

    out = ROOT / "calibration" / f"chessboard_{sx}x{sy}_{sq:g}mm.pdf"
    fig.savefig(out)
    print(f"Guardado {out}")


if __name__ == "__main__":
    main()
