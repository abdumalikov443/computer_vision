"""Exercise definitions.

Every exercise is tracked through one joint angle that gets SMALLER as the
movement goes deeper (knee for squats, elbow for push-ups and curls), so one
rep counter handles all of them:

    rest (angle > up)  ->  going down (angle < start)  ->  bottom (min angle)  ->  back above `up` = 1 rep

A rep is "full" when the lowest angle reached is <= `depth`, otherwise it is
counted as a partial rep and the coach says what to fix.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .geometry import angle_to_vertical, joint_angle

# COCO keypoint ids (YOLO11-pose)
NOSE = 0
SHOULDER = {"left": 5, "right": 6}
ELBOW = {"left": 7, "right": 8}
WRIST = {"left": 9, "right": 10}
HIP = {"left": 11, "right": 12}
KNEE = {"left": 13, "right": 14}
ANKLE = {"left": 15, "right": 16}


@dataclass
class FormCheck:
    """A rule checked on every frame of a rep; `bad(kp)` returns True when the form is off."""
    message: str
    bad: Callable[[dict], bool]
    only_near_bottom: bool = False


@dataclass
class Exercise:
    key: str
    name: str
    joint: tuple            # three keypoint maps (a, b, c); angle measured at b
    up: float               # angle above this = back at the top / rest
    start: float            # angle below this = a rep has started
    depth: float            # lowest angle must reach this for a full rep
    shallow_msg: str
    good_msg: str
    side_points: tuple      # keypoint maps used to pick the camera-facing side
    checks: List[FormCheck] = field(default_factory=list)
    # optional custom signal instead of a joint angle: metric(xy, conf) -> (value, anchor_xy) or None
    metric: Optional[Callable] = None
    unit: str = "°"
    use_midline: bool = False   # average left and right points (for lying exercises where sides get mixed)


def _pt(kp, part):
    return kp[part]


def _squat_checks():
    def chest_down(kp):
        lean = angle_to_vertical(_pt(kp, "shoulder"), _pt(kp, "hip"))
        return lean > 50

    return [FormCheck("Keep your chest up", chest_down, only_near_bottom=True)]


def _midpoint(xy, conf, a, b, min_conf=0.4):
    pts = [xy[i] for i in (a, b) if conf[i] >= min_conf]
    if not pts:
        return None
    return sum(pts) / len(pts)


def _shoulder_height(xy, conf):
    """Push-up depth: how high the shoulders are above the wrists, in % of torso length.

    Lying sideways to the camera, the model often swaps left and right arms, which
    makes the elbow angle jump. Shoulder height from the averaged left+right points
    does not care which arm is which: ~100 at the top, ~30 with the chest down.
    """
    sh = _midpoint(xy, conf, SHOULDER["left"], SHOULDER["right"])
    wr = _midpoint(xy, conf, WRIST["left"], WRIST["right"])
    hp = _midpoint(xy, conf, HIP["left"], HIP["right"])
    if sh is None or wr is None or hp is None:
        return None
    torso = float(((sh - hp) ** 2).sum() ** 0.5)
    if torso < 1:
        return None
    return (wr[1] - sh[1]) / torso * 100, sh


def _pushup_checks():
    def hips_off_line(kp):
        body = joint_angle(_pt(kp, "shoulder"), _pt(kp, "hip"), _pt(kp, "ankle"))
        return body < 150

    return [FormCheck("Keep your hips in line", hips_off_line)]


def _curl_checks():
    def elbow_swing(kp):
        upper_arm = angle_to_vertical(_pt(kp, "shoulder"), _pt(kp, "elbow"))
        return upper_arm > 35

    return [FormCheck("Keep your elbow still", elbow_swing)]


EXERCISES: Dict[str, Exercise] = {
    "squat": Exercise(
        key="squat", name="Squat",
        joint=(HIP, KNEE, ANKLE),
        up=160, start=140, depth=100,
        shallow_msg="Go deeper", good_msg="Good depth!",
        side_points=(HIP, KNEE, ANKLE),
        checks=_squat_checks(),
    ),
    "pushup": Exercise(
        key="pushup", name="Push-up",
        joint=(SHOULDER, ELBOW, WRIST),
        up=75, start=55, depth=42,
        shallow_msg="Lower your chest", good_msg="Nice push-up!",
        side_points=(SHOULDER, ELBOW, WRIST, HIP, ANKLE),
        checks=_pushup_checks(),
        metric=_shoulder_height, unit="%", use_midline=True,
    ),
    "curl": Exercise(
        key="curl", name="Bicep curl",
        joint=(SHOULDER, ELBOW, WRIST),
        up=140, start=120, depth=65,
        shallow_msg="Curl all the way up", good_msg="Full curl!",
        side_points=(SHOULDER, ELBOW, WRIST),
        checks=_curl_checks(),
    ),
}


def get_exercise(key: str) -> Exercise:
    try:
        return EXERCISES[key]
    except KeyError:
        raise ValueError(f"unknown exercise '{key}', choose from {list(EXERCISES)}") from None


def side_keypoints(xy, side: str) -> dict:
    """Named keypoints of one body side (or the left/right average for side="mid")."""
    if side == "mid":
        return {name: (xy[ids["left"]] + xy[ids["right"]]) / 2 for name, ids in
                [("shoulder", SHOULDER), ("elbow", ELBOW), ("wrist", WRIST), ("hip", HIP), ("knee", KNEE), ("ankle", ANKLE)]}
    return {
        "shoulder": xy[SHOULDER[side]], "elbow": xy[ELBOW[side]], "wrist": xy[WRIST[side]],
        "hip": xy[HIP[side]], "knee": xy[KNEE[side]], "ankle": xy[ANKLE[side]],
    }
