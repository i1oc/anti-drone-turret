"""Multi-target tracker (SORT-like, Bewley et al. 2016, with Mahalanobis gating).

Each frame:
  1. predict every track to the current time (Kalman, variable dt);
  2. cost matrix = squared Mahalanobis distance (inf if the colour differs);
  3. Hungarian assignment with the chi-square 95 % gate;
  4. second chance (cascade, as in DeepSORT/ByteTrack): established tracks
     (confirmed or lost) left unmatched are associated again with the remaining
     detections using a wider gate (relaxed_gate_factor x gate) or if the
     detection falls inside the predicted ball. A pendulum accelerates hard at
     the ends of the swing, the constant-velocity prediction lags, and the strict
     gate alone would drop the track and start a new one with a new id;
  5. matched tracks are updated with their detection;
  6. unmatched tracks count a miss; unmatched detections start new tracks.

Track life cycle:
  TENTATIVE  new track; it becomes CONFIRMED after `confirm_hits` hits.
             A tentative track that misses `tentative_max_misses` frames is
             deleted (it was probably a false detection).
  CONFIRMED  followed and shown; when it misses a frame it becomes LOST.
  LOST       coasting on the Kalman prediction (e.g. the ball passes behind
             something); back to CONFIRMED when re-detected, deleted after
             `max_misses` consecutive misses.
Only confirmed tracks get a visible ID, so noise does not consume IDs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tracking.association import CHI2_GATE_95_2D, associate
from tracking.kalman import CV, KalmanFilter

TENTATIVE = "TENTATIVE"
CONFIRMED = "CONFIRMED"
LOST = "LOST"


@dataclass
class Detection:
    x: float
    y: float
    r: float            # radius in px
    color: str = ""


@dataclass
class Track:
    kf: KalmanFilter
    color: str
    radius: float
    uid: int = 0                  # internal id (every track)
    id: int = 0                   # visible id (assigned on confirmation)
    state: str = TENTATIVE
    hits: int = 1
    misses: int = 0
    age: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    last_detection: Detection | None = None

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.kf.velocity))

    def bbox(self) -> tuple[int, int, int, int]:
        """Bounding box (x0, y0, x1, y1) around the filtered position."""
        x, y = self.position
        r = self.radius
        return int(x - r), int(y - r), int(x + r), int(y + r)


class MultiTracker:
    def __init__(self, model: str = CV, process_noise: float = 3000.0, meas_noise: float = 2.0,
                 gate: float = CHI2_GATE_95_2D, confirm_hits: int = 3, max_misses: int = 10,
                 tentative_max_misses: int = 2, radius_alpha: float = 0.3,
                 relaxed_gate_factor: float = 4.0):
        self.model = model
        self.q = process_noise ** 2         # process noise given as a std (px/s^2 for CV)
        self.r = meas_noise
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.tentative_max_misses = tentative_max_misses
        self.radius_alpha = radius_alpha
        self.relaxed_gate = relaxed_gate_factor * gate
        self.tracks: list[Track] = []
        self._next_uid = 1
        self._next_id = 1
        self._last_t: float | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "MultiTracker":
        t = cfg.get("tracker", {})
        return cls(t.get("model", CV), t.get("process_noise", 3000.0), t.get("meas_noise_px", 2.0),
                   t.get("gate", CHI2_GATE_95_2D), t.get("confirm_hits", 3), t.get("max_misses", 10),
                   t.get("tentative_max_misses", 2), 0.3, t.get("relaxed_gate_factor", 4.0))

    def reset(self) -> None:
        self.tracks = []
        self._last_t = None

    @property
    def confirmed(self) -> list[Track]:
        """Tracks to show: confirmed or temporarily lost (coasting)."""
        return [t for t in self.tracks if t.state in (CONFIRMED, LOST)]

    def _new_track(self, d: Detection) -> Track:
        tr = Track(KalmanFilter(self.model, d.x, d.y, self.q, self.r), d.color, d.r, uid=self._next_uid)
        self._next_uid += 1
        tr.history.append((d.x, d.y))
        tr.last_detection = d
        return tr

    def step(self, detections: list[Detection], t: float) -> list[Track]:
        """Process the detections of one frame taken at time t (seconds). Returns all tracks."""
        dt = 1 / 30 if self._last_t is None else max(t - self._last_t, 1e-3)
        self._last_t = t
        for tr in self.tracks:
            tr.kf.predict(dt)
            tr.age += 1

        # Cost: squared Mahalanobis distance, impossible (inf) between different colours
        cost = np.full((len(self.tracks), len(detections)), np.inf)
        for i, tr in enumerate(self.tracks):
            for j, d in enumerate(detections):
                if d.color == tr.color:
                    cost[i, j] = tr.kf.mahalanobis2((d.x, d.y))
        matches, lost_idx, new_idx = associate(cost, self.gate)

        # Second chance for established tracks with a wider gate (see module docstring)
        old = [i for i in lost_idx if self.tracks[i].state != TENTATIVE]
        if old and new_idx:
            sub = cost[np.ix_(old, new_idx)].copy()
            for a, i in enumerate(old):
                for b, j in enumerate(new_idx):
                    d = detections[j]
                    inside = np.hypot(*(self.tracks[i].position - (d.x, d.y))) <= self.tracks[i].radius
                    if inside and np.isfinite(sub[a, b]):
                        sub[a, b] = min(sub[a, b], self.gate)   # always acceptable
            m2, _, _ = associate(sub, self.relaxed_gate)
            for a, b in m2:
                matches.append((old[a], new_idx[b]))
            used_t = {old[a] for a, _ in m2}
            used_d = {new_idx[b] for _, b in m2}
            lost_idx = [i for i in lost_idx if i not in used_t]
            new_idx = [j for j in new_idx if j not in used_d]

        for i, j in matches:
            tr, d = self.tracks[i], detections[j]
            tr.kf.update((d.x, d.y))
            tr.radius += self.radius_alpha * (d.r - tr.radius)
            tr.hits += 1
            tr.misses = 0
            tr.last_detection = d
            if tr.state == TENTATIVE and tr.hits >= self.confirm_hits:
                tr.state = CONFIRMED
                tr.id = self._next_id
                self._next_id += 1
            elif tr.state == LOST:
                tr.state = CONFIRMED
            tr.history.append(tuple(tr.position))

        for i in lost_idx:
            tr = self.tracks[i]
            tr.misses += 1
            tr.last_detection = None
            if tr.state == CONFIRMED:
                tr.state = LOST
            if tr.state != TENTATIVE:
                tr.history.append(tuple(tr.position))

        self.tracks = [tr for tr in self.tracks
                       if not (tr.state == TENTATIVE and tr.misses >= self.tentative_max_misses)
                       and not (tr.state == LOST and tr.misses > self.max_misses)]
        for j in new_idx:
            self.tracks.append(self._new_track(detections[j]))
        return self.tracks
