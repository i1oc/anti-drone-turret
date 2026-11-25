"""Linear Kalman filter for a 2D point target (pure NumPy).

Two motion models, both with a variable time step dt (taken from the frame
timestamps, so a dropped frame does not break the prediction):

Constant velocity (CV), state x = [px, py, vx, vy]
    F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
    The unknown acceleration is modelled as white noise of variance q (px/s^2)^2,
    which per axis gives Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]].

Constant acceleration (CA), state x = [px, py, vx, vy, ax, ay]
    Same idea one order higher: the noise is the jerk (px/s^3)^2.

Measurement: z = H x + v with H selecting the position and R = r^2 I (px^2).

    predict:  x = F x,   P = F P F^T + Q
    update:   y = z - H x,   S = H P H^T + R,   K = P H^T S^-1
              x = x + K y,   P = (I - K H) P (I - K H)^T + K R K^T   (Joseph form, stays symmetric PSD)

A pendulum is not constant-velocity, but over one frame (33 ms) the CV model with
enough process noise follows it well; comparing CV and CA is one of the experiments.
"""
from __future__ import annotations

import numpy as np

CV = "cv"
CA = "ca"


def _block(dt: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis transition matrix and noise shape for a model of the given order (1 = CV, 2 = CA)."""
    if order == 1:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([[0.5 * dt * dt], [dt]])
    else:
        F = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        G = np.array([[dt ** 3 / 6.0], [0.5 * dt * dt], [dt]])
    return F, G @ G.T


class KalmanFilter:
    def __init__(self, model: str, x0: float, y0: float, q: float, r: float, v0_std: float = 500.0,
                 a0_std: float = 2000.0):
        if model not in (CV, CA):
            raise ValueError(f"Unknown model {model!r}")
        self.model = model
        self.order = 1 if model == CV else 2
        n = self.order + 1                       # states per axis
        self.dim = 2 * n
        # State ordered [px, py, vx, vy, (ax, ay)]
        self.x = np.zeros(self.dim)
        self.x[0], self.x[1] = x0, y0
        p_diag = [r * r, r * r, v0_std ** 2, v0_std ** 2] + ([a0_std ** 2] * 2 if n == 3 else [])
        self.P = np.diag(p_diag)
        self.q = float(q)
        self.R = np.eye(2) * r * r
        self.H = np.zeros((2, self.dim))
        self.H[0, 0] = self.H[1, 1] = 1.0

    def matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """F and Q for a time step dt, in the [px, py, vx, vy, ...] state order."""
        Fb, Gb = _block(dt, self.order)
        n = self.order + 1
        F = np.zeros((self.dim, self.dim))
        Q = np.zeros((self.dim, self.dim))
        for axis in range(2):
            idx = [axis + 2 * k for k in range(n)]   # px,vx,(ax) or py,vy,(ay)
            F[np.ix_(idx, idx)] = Fb
            Q[np.ix_(idx, idx)] = self.q * Gb
        return F, Q

    def predict(self, dt: float) -> None:
        F, Q = self.matrices(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(z, float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis2(self, z) -> float:
        """Squared Mahalanobis distance of a measurement to the predicted position."""
        y, S = self.innovation(z)
        return float(y @ np.linalg.solve(S, y))

    def update(self, z) -> None:
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    def predict_ahead(self, t: float) -> np.ndarray:
        """Predicted position t seconds ahead, without changing the filter (for lead aiming)."""
        F, _ = self.matrices(t)
        return (F @ self.x)[:2]


def make_filter(model: str, x0: float, y0: float, q: float, r: float) -> KalmanFilter:
    return KalmanFilter(model, x0, y0, q, r)
