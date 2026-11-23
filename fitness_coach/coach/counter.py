"""Rep counter: a small state machine with hysteresis on one joint angle."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Rep:
    index: int
    start_frame: int
    end_frame: int
    min_angle: float
    full: bool
    issues: List[str] = field(default_factory=list)


class RepCounter:
    """Counts reps from a stream of (smoothed) joint angles.

    States: "up" (resting at the top) and "down" (inside a rep).
    up -> down when angle < start; down -> up when angle > up (rep finished).
    The gap between `start` and `up` is the hysteresis that stops jitter from
    being counted as extra reps.
    """

    def __init__(self, up: float, start: float, depth: float, min_frames: int = 4):
        if not depth < start < up:
            raise ValueError("thresholds must satisfy depth < start < up")
        self.up, self.start, self.depth = up, start, depth
        self.min_frames = min_frames
        self.state = "up"
        self.reps: List[Rep] = []
        self._rep_start = 0
        self._min = float("inf")
        self._issues: List[str] = []

    @property
    def count(self) -> int:
        return len(self.reps)

    @property
    def full_count(self) -> int:
        return sum(r.full for r in self.reps)

    @property
    def in_rep(self) -> bool:
        return self.state == "down"

    @property
    def current_min(self) -> Optional[float]:
        return self._min if self.in_rep else None

    def flag(self, message: str):
        """Record a form issue for the rep in progress."""
        if self.in_rep and message not in self._issues:
            self._issues.append(message)

    def update(self, angle: Optional[float], frame: int) -> Optional[Rep]:
        """Feed one angle; returns the finished Rep on the frame a rep completes."""
        if angle is None:
            return None
        angle = float(angle)
        if self.state == "up":
            if angle < self.start:
                self.state = "down"
                self._rep_start = frame
                self._min = angle
                self._issues = []
            return None
        # state == "down"
        self._min = min(self._min, angle)
        if angle > self.up:
            self.state = "up"
            if frame - self._rep_start < self.min_frames:
                return None  # too short to be a real rep: jitter
            rep = Rep(
                index=len(self.reps) + 1,
                start_frame=self._rep_start,
                end_frame=frame,
                min_angle=round(float(self._min), 1),
                full=bool(self._min <= self.depth),
                issues=list(self._issues),
            )
            self.reps.append(rep)
            return rep
        return None
