"""Drawing: skeleton, joint angle and the coach HUD on a video frame."""
import cv2
import numpy as np

# skeleton edges (COCO ids), drawn on both sides
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]
LEFT_IDS = {5, 7, 9, 11, 13, 15}

WHITE = (255, 255, 255)
DARK = (24, 22, 20)
BLUE = (232, 71, 35)      # BGR of #2347e8
GREEN = (94, 197, 34)
ORANGE = (0, 150, 255)
GREY = (170, 170, 170)


def _p(pt):
    return int(round(pt[0])), int(round(pt[1]))


def draw_skeleton(img, xy, conf, active_side, joint_ids, s, min_conf=0.4, both_sides=False):
    if both_sides:
        side_ids = set(range(5, 17))
        joint_ids = set()
    else:
        side_ids = LEFT_IDS if active_side == "left" else {i for i in range(5, 17)} - LEFT_IDS
    for a, b in EDGES:
        if conf[a] < min_conf or conf[b] < min_conf:
            continue
        on_side = a in side_ids and b in side_ids
        color = BLUE if on_side else GREY
        cv2.line(img, _p(xy[a]), _p(xy[b]), color, max(2, int(4 * s)) if on_side else max(1, int(2 * s)), cv2.LINE_AA)
    for i in range(5, 17):
        if conf[i] < min_conf:
            continue
        r = int(6 * s) if i in joint_ids else int(4 * s)
        cv2.circle(img, _p(xy[i]), r, WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, _p(xy[i]), r, BLUE if i in side_ids else GREY, max(1, int(2 * s)), cv2.LINE_AA)


def draw_joint_angle(img, center, angle, s, unit="°"):
    if angle is None:
        return
    x, y = _p(center)
    text = f"{int(round(angle))}" + ("%" if unit == "%" else "")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8 * s, max(1, int(2 * s)))
    x0, y0 = x + int(14 * s), y - int(14 * s)
    cv2.rectangle(img, (x0 - 6, y0 - th - 6), (x0 + tw + 14, y0 + 6), DARK, -1)
    cv2.putText(img, text, (x0, y0), cv2.FONT_HERSHEY_DUPLEX, 0.8 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    if unit != "%":  # degree sign (OpenCV fonts have no °)
        cv2.circle(img, (x0 + tw + 7, y0 - th + 2), max(2, int(3 * s)), WHITE, 1, cv2.LINE_AA)


def _alpha_rect(img, p1, p2, color, alpha):
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_hud(img, exercise, reps, full, angle, progress, message, msg_good, s):
    """Top-left panel: exercise, rep count, full/partial, depth bar; bottom banner: feedback."""
    h, w = img.shape[:2]
    pw, ph = int(250 * s), int(178 * s)
    m = int(16 * s)
    _alpha_rect(img, (m, m), (m + pw, m + ph), DARK, 0.78)
    cv2.rectangle(img, (m, m), (m + int(6 * s), m + ph), BLUE, -1)
    x = m + int(20 * s)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, exercise.upper(), (x, m + int(30 * s)), font, 0.62 * s, (200, 200, 200), max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.putText(img, str(reps), (x, m + int(98 * s)), font, 2.4 * s, WHITE, max(2, int(4 * s)), cv2.LINE_AA)
    cv2.putText(img, "REPS", (x + int(84 * s) if reps < 10 else x + int(130 * s), m + int(98 * s)), font, 0.6 * s, (200, 200, 200), max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.putText(img, f"full {full}  /  partial {reps - full}", (x, m + int(128 * s)), font, 0.55 * s, (210, 210, 210), max(1, int(1.2 * s)), cv2.LINE_AA)
    # depth bar
    bx1, by = x, m + int(150 * s)
    bx2 = m + pw - int(20 * s)
    cv2.rectangle(img, (bx1, by), (bx2, by + int(10 * s)), (70, 70, 70), -1)
    if progress is not None:
        fill = bx1 + int((bx2 - bx1) * max(0.0, min(1.0, progress)))
        cv2.rectangle(img, (bx1, by), (fill, by + int(10 * s)), GREEN if progress >= 1 else BLUE, -1)
    cv2.putText(img, "depth", (bx2 - int(46 * s), by - int(6 * s)), font, 0.42 * s, (170, 170, 170), 1, cv2.LINE_AA)

    if message:
        color = GREEN if msg_good else ORANGE
        (tw, th), _ = cv2.getTextSize(message, font, 0.9 * s, max(1, int(2 * s)))
        bw, bh = tw + int(40 * s), th + int(28 * s)
        bx, byy = (w - bw) // 2, h - bh - int(24 * s)
        _alpha_rect(img, (bx, byy), (bx + bw, byy + bh), DARK, 0.8)
        cv2.rectangle(img, (bx, byy), (bx + int(6 * s), byy + bh), color, -1)
        cv2.putText(img, message, (bx + int(22 * s), byy + bh - int(14 * s)), font, 0.9 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
