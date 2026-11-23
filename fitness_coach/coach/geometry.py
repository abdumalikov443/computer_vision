"""Small geometry helpers for 2D keypoints."""
import math

import numpy as np


def joint_angle(a, b, c):
    """Angle ABC in degrees (0..180), measured at point b."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return float("nan")
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(math.degrees(math.acos(cos)))


def angle_to_vertical(top, bottom):
    """How far the segment bottom->top leans from straight up, in degrees (0 = upright)."""
    dx, dy = top[0] - bottom[0], top[1] - bottom[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float("nan")
    # image y grows downward, so "up" is (0, -1)
    cos = -dy / math.hypot(dx, dy)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))


class EMA:
    """Exponential moving average that ignores missing values (NaN)."""

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return self.value
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value
