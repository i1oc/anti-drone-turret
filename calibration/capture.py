"""Captura de imágenes del tablero para la calibración offline.

Muestra la cámara en vivo, detecta el tablero en cada frame y guarda la imagen
(sin overlay) al pulsar ESPACIO. Un mapa de cobertura en verde indica qué zonas
de la imagen ya tienen tablero: hay que cubrir también bordes y esquinas, que es
donde más se nota la distorsión.

Consejos: 20-30 imágenes; distintas distancias; inclinaciones de hasta ~45° en
ambos ejes; tablero quieto al guardar (sin desenfoque de movimiento); misma
resolución que el sistema (config.yaml).

Uso:
    python calibration/capture.py [--camera 1]

Teclas:
    ESPACIO  guardar imagen (solo si se detecta el tablero)
    z        deshacer: borra la última imagen guardada en esta sesión
    q / ESC  salir
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from calib_utils import find_corners, pattern_size  # noqa: E402
from camera import ThreadedCamera  # noqa: E402
from fps import FPSCounter, draw_stats  # noqa: E402
from settings import load_config, project_path  # noqa: E402

MIN_SHIFT_PX = 10.0  # una imagen casi idéntica a otra ya guardada no aporta información


def outer_quad(corners: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    """Las 4 esquinas exteriores de la rejilla de esquinas detectadas."""
    cols, rows = pattern
    g = corners.reshape(rows, cols, 2)
    return np.array([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]], np.int32)


def add_coverage(coverage: np.ndarray, corners: np.ndarray, pattern: tuple[int, int]) -> None:
    """Suma 1 en la zona que ocupa el tablero (para ver cuántas veces se ha cubierto cada zona)."""
    m = np.zeros(coverage.shape, np.uint8)
    cv2.fillConvexPoly(m, outer_quad(corners, pattern), 1)
    coverage += m


def main() -> None:
    p = argparse.ArgumentParser(description="Captura de imágenes de calibración")
    p.add_argument("--camera", type=int, help="índice de la cámara (sobrescribe la config)")
    p.add_argument("--out", type=Path, help="carpeta de salida (por defecto la de la config)")
    args = p.parse_args()

    cfg = load_config()
    cam_cfg, cal = cfg["camera"], cfg["calibration"]
    pattern = pattern_size(cal["squares_x"], cal["squares_y"])
    out_dir = args.out or project_path(cal["images_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    index = args.camera if args.camera is not None else cam_cfg["index"]

    with ThreadedCamera(index, cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"],
                        cam_cfg.get("autofocus")) as cam:
        w, h = cam.resolution
        if (w, h) != (cam_cfg["width"], cam_cfg["height"]):
            print(f"AVISO: la cámara da {w}x{h} y la config pide "
                  f"{cam_cfg['width']}x{cam_cfg['height']}. Hay que calibrar a la resolución de uso.")

        # Cobertura acumulada (incluye las imágenes que ya hubiera en la carpeta)
        coverage = np.zeros((h, w), np.float32)
        saved: list[np.ndarray] = []
        existing = sorted(out_dir.glob("*.png"))
        for f in existing:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (h, w):
                continue
            c = find_corners(gray, pattern, subpix=False)
            if c is not None:
                add_coverage(coverage, c, pattern)
                saved.append(c.reshape(-1, 2))
        print(f"{len(existing)} imágenes previas en {out_dir}. ESPACIO guarda, z deshace, q sale.")

        session_files: list[Path] = []
        session_corners: list[np.ndarray] = []
        fps = FPSCounter()
        last_id, msg, msg_t = -1, "", 0.0
        window = "Captura de calibracion"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            frame_id, frame = cam.read()
            if frame_id == last_id:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue
            last_id = frame_id

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, pattern, fast=True, subpix=False)

            out = frame.copy()
            if coverage.any():
                tint = np.zeros_like(out)
                tint[..., 1] = np.clip(coverage * 60 + 60, 0, 255).astype(np.uint8)
                mask = coverage > 0
                out[mask] = cv2.addWeighted(out, 0.65, tint, 0.35, 0)[mask]
            if corners is not None:
                cv2.drawChessboardCorners(out, pattern, corners, True)
            pct = 100.0 * np.count_nonzero(coverage) / coverage.size
            n_total = len(saved)
            lines = [f"FPS: {fps.tick():.1f}", f"Imagenes: {n_total}  Cobertura: {pct:.0f}%",
                     "TABLERO OK" if corners is not None else "sin tablero"]
            if msg and time.time() - msg_t < 2.0:
                lines.append(msg)
            draw_stats(out, lines)
            cv2.imshow(window, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if corners is None:
                    msg = "No se ve el tablero entero"
                else:
                    pts = corners.reshape(-1, 2)
                    shift = min((np.linalg.norm(pts - s, axis=1).mean() for s in saved), default=np.inf)
                    if shift < MIN_SHIFT_PX:
                        msg = "Casi igual que otra: mueve o inclina el tablero"
                    else:
                        name = out_dir / f"calib_{time.strftime('%Y%m%d_%H%M%S')}_{frame_id}.png"
                        cv2.imwrite(str(name), frame)  # imagen limpia, sin overlay
                        add_coverage(coverage, corners, pattern)
                        saved.append(pts)
                        session_files.append(name)
                        session_corners.append(pts)
                        msg = f"Guardada {name.name}"
                        print(msg)
                msg_t = time.time()
            if key == ord("z") and session_files:
                f = session_files.pop()
                pts = session_corners.pop()
                f.unlink(missing_ok=True)
                saved = [s for s in saved if s is not pts]
                # Reconstruir la cobertura sin la imagen borrada
                coverage[:] = 0
                for s in saved:
                    add_coverage(coverage, s, pattern)
                msg, msg_t = f"Borrada {f.name}", time.time()
                print(msg)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
