"""Video capture in a separate thread.

cap.read() blocks until the next frame arrives. If it is called in the main
loop, processing waits for the camera. With a dedicated thread, the main loop
always gets the most recent frame instantly.
"""
from __future__ import annotations

import platform
import threading
import time

import cv2


class ThreadedCamera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30,
                 autofocus: bool | None = None, exposure: float | None = None):
        # On Windows, DirectShow opens the camera faster and without warnings
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}. Try another index.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # do not queue old frames
        if autofocus is not None:
            # With autofocus on, the focal length changes and the calibration is no longer valid.
            # If the camera does not support it, set() returns False and nothing happens.
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if exposure is not None:
            # Fixed exposure: auto-exposure burns bright objects against a dark background and
            # changes the colours over time. 0.25 = manual mode in DirectShow (0.75 = auto).
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("The camera opened but returns no images.")

        self._frame = frame
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    @property
    def resolution(self) -> tuple[int, int]:
        h, w = self._frame.shape[:2]
        return w, h

    def _update(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self):
        """Return (id, frame). The id tells whether the frame is new."""
        with self._lock:
            return self._frame_id, self._frame

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
