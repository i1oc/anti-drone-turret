# Torreta antidrón virtual con objetivos reales

Proyecto final de **Visión por Ordenador I** · 3º Ingeniería Matemática · Universidad Pontificia Comillas ICAI · curso 2026/27.

Una cámara real fija vigila una zona por la que se mueven objetivos físicos (pelotas de colores colgadas como péndulos) que hacen de "drones". El sistema permanece **bloqueado** hasta que el operador muestra a la cámara una secuencia de 4 tarjetas impresas en el orden correcto. Una vez **armado**, detecta y sigue cada objetivo con un filtro de Kalman, y una torreta **virtual** dibujada en pantalla gira, apunta con adelanto y dispara un proyectil virtual cuando prevé un impacto.

Todo es visión por ordenador clásica con Python y OpenCV, sin deep learning y sin hardware adicional.

> **Estado:** cámara en tiempo real con FPS y calibración offline con corrección de distorsión.

## Equipo

| Integrante | Módulo principal |
| --- | --- |
| _Nombre Apellido_ | Calibración y geometría |
| _Nombre Apellido_ | Tarjetas y decodificador |
| _Nombre Apellido_ | Detección y tracking |
| _Nombre Apellido_ | Integración, evaluación y documentación |

Grupo: _A / B_

## Instalación

Requisitos: **Python 3.9** y Git. Las versiones de las librerías son las del entorno de prácticas del curso ([icai-voi/voi-26-27](https://github.com/icai-voi/voi-26-27)); OpenCV 4.8 y NumPy 1.26 no tienen versión para Python 3.13 o superior.

```bash
git clone https://github.com/<usuario>/torreta-antidron.git
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

## Uso

```bash
python src/main.py            # cámara definida en config/config.yaml
python src/main.py --camera 1 # otra cámara
pytest                        # tests
```

### Calibración de la cámara (offline, una vez)

Tablero de 11×8 cuadros de 20 mm (10×7 esquinas interiores); las medidas están en `config/config.yaml`.

```bash
python calibration/make_chessboard.py   # PDF A4: imprimir al 100 % y comprobar la regla de 50 mm
python calibration/capture.py           # ESPACIO guarda, z deshace, q sale (20-30 imágenes)
python calibration/calibrate.py         # K, distorsión, RMS y RMS de validación -> calibration/calib.yaml
```

`calibrate.py` reserva un 20 % de las imágenes para medir el **RMS de validación** (error de
reproyección en imágenes no usadas), compara el modelo de distorsión de 4 y 5 coeficientes y guarda
las figuras del informe en `docs/figures/calibration/`. Con `--no-curve` omite la curva de error frente al
número de imágenes, que es la parte lenta.

| Tecla | Acción |
| --- | --- |
| `q` / `Esc` | Salir |
| `s` | Guardar el frame actual en `data/snapshots/` |
| `u` | Activar/desactivar la corrección de distorsión (requiere `calibration/calib.yaml`) |

## Estructura del repositorio

```text
torreta-antidron/
├── src/                # código del sistema
│   ├── main.py         # punto de entrada y bucle en tiempo real
│   ├── camera.py       # captura en un hilo aparte
│   ├── camera_model.py # corrección de distorsión, píxel -> ángulos, profundidad
│   ├── calib_utils.py  # detección del tablero, calibración, RMS de validación
│   ├── settings.py     # rutas y lectura de la config
│   └── fps.py          # FPS y overlay de rendimiento
├── config/config.yaml  # parámetros (cámara, umbrales, código de armado...)
├── calibration/        # scripts de calibración, imágenes del tablero y calib.yaml
├── tests/              # tests con pytest
├── docs/figures/       # figuras del informe
└── data/               # vídeos y capturas de prueba (no se suben al repo)
```

## Hoja de ruta

- [x] Lectura de la cámara en tiempo real con FPS
- [x] Calibración offline (K, distorsión, RMS) y corrección de distorsión (RMS 0.22 px, validación 0.23 px)
- [ ] Detector de tarjetas de armado
- [ ] Decodificador de secuencia (máquina de estados)
- [ ] Detección y tracking multiobjetivo (Kalman + asignación húngara)
- [ ] Control de tiro: torreta virtual, apuntado con adelanto, IFF, radar
- [ ] Evaluación cuantitativa y vídeo demo

## Uso de asistentes de IA

Se ha utilizado Claude (Anthropic) como asistente de diseño y programación. Todo el código ha sido revisado, probado y entendido por el equipo.
