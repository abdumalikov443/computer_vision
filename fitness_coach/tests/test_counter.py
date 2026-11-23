"""Rep counter tests on synthetic angle curves (no model or video needed)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coach.counter import RepCounter  # noqa: E402
from coach.geometry import EMA, angle_to_vertical, joint_angle  # noqa: E402


def squat_wave(reps, bottom, top=175, frames_per_rep=40):
    """Cosine wave: standing (top) -> bottom -> standing, `reps` times."""
    out = []
    for _ in range(reps):
        for i in range(frames_per_rep):
            t = i / frames_per_rep
            out.append(top - (top - bottom) * (1 - math.cos(2 * math.pi * t)) / 2)
    out.append(top)
    return out


def run(counter, angles):
    for i, a in enumerate(angles):
        counter.update(a, i)
    return counter


def test_counts_full_reps():
    c = run(RepCounter(up=160, start=140, depth=100), squat_wave(5, bottom=75))
    assert c.count == 5 and c.full_count == 5


def test_shallow_reps_are_partial():
    c = run(RepCounter(up=160, start=140, depth=100), squat_wave(3, bottom=115))
    assert c.count == 3 and c.full_count == 0
    assert all(r.min_angle > 100 for r in c.reps)


def test_too_shallow_is_not_a_rep():
    # never goes below `start`: just shifting weight, not a squat
    c = run(RepCounter(up=160, start=140, depth=100), squat_wave(4, bottom=150))
    assert c.count == 0


def test_jitter_near_threshold_is_ignored():
    # hovering around `start` with noise must not create reps (hysteresis)
    angles = [175] * 10 + [141, 139, 142, 138, 141, 139, 143] * 5 + [175] * 10
    c = run(RepCounter(up=160, start=140, depth=100), angles)
    assert c.count <= 1


def test_missing_frames_do_not_break_counting():
    angles = squat_wave(3, bottom=80)
    angles = [None if i % 7 == 0 else a for i, a in enumerate(angles)]
    c = run(RepCounter(up=160, start=140, depth=100), angles)
    assert c.count == 3


def test_flagged_issue_is_attached_to_rep():
    c = RepCounter(up=160, start=140, depth=100)
    for i, a in enumerate(squat_wave(1, bottom=80)):
        c.update(a, i)
        if c.in_rep and a < 100:
            c.flag("Keep your chest up")
    assert c.reps[0].issues == ["Keep your chest up"]


def test_bad_thresholds_rejected():
    with pytest.raises(ValueError):
        RepCounter(up=100, start=140, depth=90)


def test_geometry():
    assert joint_angle((0, 0), (1, 0), (2, 0)) == pytest.approx(180)
    assert joint_angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90)
    assert angle_to_vertical((0, 0), (0, 10)) == pytest.approx(0)    # straight up in image coords
    assert angle_to_vertical((10, 10), (0, 10)) == pytest.approx(90)  # horizontal
    e = EMA(0.5)
    assert e.update(10) == 10 and e.update(20) == 15 and e.update(float("nan")) == 15
