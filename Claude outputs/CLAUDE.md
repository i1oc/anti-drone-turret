# CLAUDE.md — Torreta antidrón virtual con objetivos reales

Contexto para Claude Code. Responde siempre en **español**.

## Qué es

Proyecto final de **Visión por Ordenador I** (3º Ingeniería Matemática, Comillas ICAI, curso 2026/27).
Una webcam fija vigila una zona por la que se mueven pelotas de colores colgadas como péndulos ("drones").
El sistema arranca **bloqueado**; el operador muestra 4 tarjetas impresas en el orden correcto y el sistema
se **arma**: detecta y sigue cada objetivo (Kalman + asignación húngara) y una torreta **virtual** dibujada
en pantalla gira, apunta con adelanto y dispara un proyectil virtual. Sin hardware adicional.

- **Equipo (grupo A):** Hugo Lara Rodríguez, Pablo Santiago Hernández Muñoz
- **Repo:** https://github.com/i1oc/torreta-antidron (público)
- **Identidad git:** los commits deben salir como `i1oc <i1oc@proton.me>` (configurado en local). Nunca como `ppx005`.
- **Entrega final y presentación: 2 de diciembre de 2026.** Propuesta ya entregada.

## Requisitos obligatorios (guion oficial)

- Cámara física real como entrada. Visión **clásica**: nada de deep learning sin aprobación.
- Calibración **offline** con patrón; el informe incluye K, distorsión y **RMS**; incluye corrección de distorsión.
- Sistema de seguridad: detector de patrones sencillos (se valora el método **y el diseño** de los patrones) +
  decodificador que memoriza hasta 4 patrones, permite/bloquea el paso y puede resetear la secuencia.
- Tras la secuencia correcta, tracker con **bounding box** que sigue la zona de interés.
- Salida de vídeo en **tiempo real con los FPS dibujados en cada frame**. Se valora código optimizado.
- Las ampliaciones suman.

**Nota N (sobre 10):** calibración 1 · detector de patrones 2 · decodificador 2 · tracker 2 · vídeo en tiempo real 1 · ampliaciones 2.
**Factor F:** informe 0.50 (máx. 5 páginas, con bibliografía y declaración de uso de IA) · repositorio 0.25 (README completo) · vídeo 0.25 (demo con póster/PPT, FPS visibles). Nota = N × F.

Objetivo del equipo: **el 10**. Cada módulo debe respaldarse con una medida en Resultados (ver "Experimentos").

## Entorno (no cambiar sin motivo)

- Windows, **Python 3.9** en `.venv` (`py -3.9 -m venv .venv`). Versiones fijadas en `requirements.txt` =
  entorno `voi-lab` de las prácticas (numpy 1.26.0, opencv-python 4.8.0.76, scipy 1.13.1, matplotlib 3.9.4…) + pytest.
- Código compatible con 3.9: `from __future__ import annotations` si se usan `list[str]` o `X | None`; sin `match`.
- No añadir dependencias nuevas sin preguntar.
- Cámara: **webcam estándar** → modelo pinhole (`cv2.calibrateCamera`), resolución de trabajo **640×480**, sin espejo.

## Estado actual del repo

```text
src/main.py          bucle en tiempo real: FPS, 's' guarda frame, 'u' corrige distorsión
src/camera.py        ThreadedCamera: captura en un hilo (CAP_DSHOW), autofocus configurable
src/fps.py           FPSCounter (EMA) y draw_stats (overlay)
src/settings.py      ROOT, load_config, project_path
src/calib_utils.py   board_points, find_corners, calibrate, RMS de validación, YAML
src/camera_model.py  CameraModel: undistort (remap), pixel_to_angles, depth_from_diameter
calibration/         make_chessboard.py, capture.py, calibrate.py, calib.yaml, images/ (47 fotos)
                     HECHO: RMS 0.220 px, validación 0.229 px, fx≈495, 4 coef. (k3 no aporta), remap ≈1 ms, 30 FPS
src/card_symbols.py  diseño de las 8 tarjetas (marco 0.14, margen 0.10, símbolo r 0.27) y render_card
src/card_detector.py CardDetector: umbral adaptativo -> quads con hueco -> homografía 200x200 -> Otsu ->
                     rasgos (pieces, holes, fill, solidity, vertices, regularity, tilt, elongation) -> reglas.
                     Polígonos por nº de vértices (el relleno sube con desenfoque: el rombo/DIAMOND fallaba en pantalla real).
                     'c' activa, 'd' guarda etapas + features.yaml. Sintético: 800/800 hasta 40°; 96-97 % a 55°,
                     tarjeta pequeña y desenfoque, 0 confusiones, 0 falsos positivos en 300 fondos; ~6 ms
cards/make_cards.py  cards_90mm.pdf: 4 tarjetas de 90 mm por A4
config/config.yaml   cámara, display, calibration, cards
tests/               pytest (pytest.ini: pythonpath=src); synth.py = cámara virtual para tests
docs/figures/calibration/ · data/ (ignorado por git)
.gitattributes / .gitignore (.venv y data/ excluidos)
```

## Diseño acordado por módulo

1. **Calibración** — scripts en `calibration/`: `make_chessboard.py` (PDF A4, tablero de 11×8 cuadros = 10×7 esquinas
   interiores, cuadro 20 mm, regla de 50 mm para verificar escala), `capture.py` (detección en vivo, guardar con espacio, mapa de cobertura),
   `calibrate.py` (findChessboardCorners + cornerSubPix + calibrateCamera; RMS y **RMS de validación** con imágenes
   no usadas; error por imagen; guarda `calibration/calib.yaml`; figuras para el informe).
   Runtime en `src/camera_model.py`: carga YAML, mapas `initUndistortRectifyMap` + `remap` (tecla `u` en main),
   `undistortPoints` para centroides, píxel→ángulos con x=(u−cx)/fx, y=(v−cy)/fy:
   `az = atan(x)`, `el = −atan(y/√(1+x²))`; profundidad `Z ≈ f·D/d` con pelota de diámetro conocido.
   Reutilizar código de la práctica 2.
2. **Detector de tarjetas** — tarjeta = marco negro grueso + símbolo interior. 8 clases: círculo, diana (2 círculos
   concéntricos), triángulo, cuadrado girado, pentágono, 3 barras, 4 puntos, aspa CANCELAR. Pipeline: gris → blur →
   umbral adaptativo/Otsu → `findContours(RETR_TREE)` → cuadrilátero con hueco → homografía a 200×200 → rasgos
   invariantes (nº vértices `approxPolyDP`, anidamiento, circularidad, Hough) → clase o DESCONOCIDO. ROI central.
   `make_cards.py` genera las tarjetas imprimibles.
3. **Decodificador** — máquina de estados LOCKED → ENTERING(1..4) → VERIFY → ARMED (+ PENALTY).
   Anti-rebote: N≈10 frames estables y exigir "sin tarjeta" entre lecturas. Reset: CANCELAR, timeout 8 s, tecla `r`.
   3 fallos → bloqueo 10/20/40 s. Código en `config.yaml`. Tests con secuencias simuladas (sin cámara).
4. **Tracker** — HSV `inRange` por color (rojo con 2 rangos) → morfología → contornos → área + circularidad > 0.7 →
   `minEnclosingCircle`; MOG2 solo como confirmación (para no detectar la mano). `cv2.KalmanFilter` por objetivo
   (velocidad constante; comparar con aceleración constante). Asociación: Mahalanobis + puerta χ² 95 % +
   `scipy.optimize.linear_sum_assignment`. Pistas provisional → confirmada (3) → perdida (10). Box, ID, velocidad, estela.
5. **Control de tiro (ampliaciones)** — torreta con velocidad angular máxima; interceptación por punto fijo con la
   predicción del Kalman y la profundidad; disparo si error previsto < tolerancia y sin amigos en la línea de tiro;
   IFF por color (verde = amigo); priorización de amenazas; vista radar; **objetivos sintéticos** con trayectoria
   conocida superpuestos al vídeo real para medir el tracker y el % de impactos.
6. **Tiempo real** — FPS + ms por módulo en pantalla; solo se ejecuta lo del modo activo; ROIs; mapas precalculados;
   grabación de la demo con `VideoWriter` en otro hilo. Objetivo ≥ 30 fps.

## Convenciones

- **Repo en inglés**: comentarios, docstrings, README, textos en pantalla, identificadores, nombres de clases
  (`CIRCLE`, `CANCEL`, `UNKNOWN`...), claves de rasgos, nombres de archivos generados y mensajes de commit.
  Los estados del decodificador también irán en inglés (LOCKED, ENTERING, VERIFY, ARMED, PENALTY).
  El chat con el usuario sigue en español. Commits pequeños por módulo.
- Parámetros en `config/config.yaml`, no hardcodeados.
- `tracking/` y `fire_control/`: **funciones/clases puras sin dependencias de OpenCV y con tests**. Se portarán a C#
  para un juego en Unity de otra asignatura (Paradigmas) a partir del 3 dic.
- Cada etapa debe poder guardar su imagen intermedia (original, corregida, HSV, máscaras, contornos, tarjeta
  rectificada) para la figura obligatoria de "secuencia de transformaciones" del informe.
- Antes de un cambio grande, explica el plan y espera confirmación.
- **No borres archivos del usuario ni hagas operaciones destructivas de git** (`reset --hard`, `push --force`,
  reescribir historia) sin pedir permiso explícito.

## Experimentos para Resultados (objetivo de nota 10)

| Módulo | Medida |
| --- | --- |
| Calibración | RMS de calibración y de validación (< 0.5 px); error angular frente a marcas medidas en la pared (< 1°) |
| Tarjetas | Matriz de confusión con distintas luces, distancias y giros; distancia/inclinación máximas |
| Decodificador | Falsas aceptaciones (0) y tiempo medio de desbloqueo |
| Tracker | RMSE, IoU y cambios de ID con objetivos sintéticos; Kalman vs detección bruta vs CSRT/KCF; péndulo real |
| Control de tiro | % de impactos con y sin adelanto a varias velocidades |
| Tiempo real | FPS y ms por módulo antes/después de optimizar |

## Calendario (grupo A)

Sistema mínimo de principio a fin (seguridad + tracker) lo antes posible · 26 nov cierre de funcionalidades ·
26–28 nov experimentos · 29 nov–2 dic informe (plantilla LaTeX del repo del curso), vídeo y README.

## Referencias

- Repo del curso con las prácticas y la plantilla LaTeX: https://github.com/icai-voi/voi-26-27
  (Lab 2 calibración, Lab 3 procesado, Lab 4 Harris/Hough/SIFT, Lab 5 MOG2/flujo óptico/Kalman; `Lab_Project/src/test.py`).
- Bibliografía prevista: Zhang (2000) calibración; Suzuki & Abe (1985) contornos; Douglas-Peucker (1973); Otsu (1979);
  Duda & Hart (1972) Hough; Zivkovic (2004/2006) MOG2; Kalman (1960); Kuhn (1955) asignación húngara;
  Bewley et al. (2016) SORT; Szeliski, *Computer Vision: Algorithms and Applications*.
