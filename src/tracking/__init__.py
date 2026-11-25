"""Multi-target tracking core: Kalman filters, data association and track management.

Pure Python + NumPy (and SciPy for the Hungarian algorithm): no OpenCV, so it can
be unit-tested without images and ported to C# (Unity) later.
"""
from tracking.kalman import KalmanFilter, make_filter
from tracking.association import CHI2_GATE_95_2D, associate
from tracking.tracker import CONFIRMED, LOST, TENTATIVE, Detection, MultiTracker, Track

__all__ = ["KalmanFilter", "make_filter", "CHI2_GATE_95_2D", "associate", "Detection", "Track",
           "MultiTracker", "TENTATIVE", "CONFIRMED", "LOST"]
