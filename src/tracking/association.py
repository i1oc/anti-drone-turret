"""Data association: which detection belongs to which track (global nearest neighbour).

Cost of pairing track i with detection j = squared Mahalanobis distance between
the detection and the track's PREDICTED position, using the innovation
covariance S = H P H^T + R. Unlike the Euclidean distance it accounts for the
uncertainty of each track: a fast or long-unseen track has a larger S and may
match farther away.

Gate: pairs with d^2 above the chi-square 95 % quantile for 2 degrees of freedom
(5.991) are impossible: under the model, a correct detection falls outside the
gate only 5 % of the time. Pairs of different colour are impossible too.

The optimal one-to-one assignment (minimum total cost) is found with the
Hungarian algorithm (Kuhn 1955; scipy.optimize.linear_sum_assignment).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_GATE_95_2D = 5.991  # chi2.ppf(0.95, df=2)
_BIG = 1e6


def associate(cost: np.ndarray, gate: float = CHI2_GATE_95_2D):
    """Optimal assignment on a cost matrix (tracks x detections), with gating.

    Returns (matches [(i, j)], unmatched_tracks [i], unmatched_detections [j]).
    Entries that are np.inf or above the gate are never matched.
    """
    n_t, n_d = cost.shape
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))
    c = np.where(np.isfinite(cost) & (cost <= gate), cost, _BIG)
    rows, cols = linear_sum_assignment(c)
    matches = [(int(i), int(j)) for i, j in zip(rows, cols) if c[i, j] < _BIG]
    mt = {i for i, _ in matches}
    md = {j for _, j in matches}
    return matches, [i for i in range(n_t) if i not in mt], [j for j in range(n_d) if j not in md]
