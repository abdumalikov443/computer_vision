"""Run YOLO11-pose over a video, count reps, check form, write an annotated video."""
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Optional

import cv2
import numpy as np

from .counter import RepCounter
from .exercises import get_exercise, side_keypoints
from .geometry import EMA, joint_angle
from . import render

MIN_CONF = 0.4          # keypoint confidence needed to trust a point
BAD_STREAK = 3          # frames a form problem must last before it is flagged
MESSAGE_FRAMES = 45     # how long feedback stays on screen after a rep

_MODEL_CACHE = {}


def load_model(weights: str = "yolo11n-pose.pt"):
    from ultralytics import YOLO
    if weights not in _MODEL_CACHE:
        _MODEL_CACHE[weights] = YOLO(weights)
    return _MODEL_CACHE[weights]


def _main_person(result):
    """Keypoints of the largest detected person, or None."""
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(areas))
    xy = result.keypoints.xy[i].cpu().numpy()
    conf = result.keypoints.conf
    conf = conf[i].cpu().numpy() if conf is not None else np.ones(len(xy))
    return xy, conf


def _to_h264(src, dst):
    """Re-encode to H.264 so browsers (and Gradio) can play the file."""
    if shutil.which("ffmpeg") is None:
        shutil.move(src, dst)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
        check=True,
    )
    os.remove(src)


def analyze_video(
    video_path: str,
    exercise: str,
    out_dir: str,
    weights: str = "yolo11n-pose.pt",
    imgsz: int = 640,
    progress: Optional[Callable[[float], None]] = None,
) -> dict:
    ex = get_exercise(exercise)
    model = load_model(weights)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(out_dir, f"{stem}_{ex.key}_coach.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = max(0.6, min(w, h) / 720)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = RepCounter(ex.up, ex.start, ex.depth)
    angle_ema = EMA(0.45)
    side_conf = {"left": EMA(0.2), "right": EMA(0.2)}
    streak = {c.message: 0 for c in ex.checks}
    message, msg_good, msg_left = "", True, 0
    series = []  # (frame, raw angle, smoothed angle)
    t0 = time.time()
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=imgsz, conf=0.35, verbose=False)[0]
        person = _main_person(result)
        raw = smooth = None
        side = "left"
        if person is not None:
            xy, conf = person
            for sd in ("left", "right"):
                side_conf[sd].update(float(np.mean([conf[p[sd]] for p in ex.side_points])))
            side = max(side_conf, key=lambda k: side_conf[k].value or 0)
            a, b, c = (p[side] for p in ex.joint)
            anchor = xy[b]
            if ex.metric is not None:
                got = ex.metric(xy, conf)
                if got is not None:
                    raw, anchor = got
                    smooth = angle_ema.update(raw)
            elif min(conf[a], conf[b], conf[c]) >= MIN_CONF:
                raw = joint_angle(xy[a], xy[b], xy[c])
                smooth = angle_ema.update(raw)

            # form checks while a rep is in progress
            live_issue = None
            if counter.in_rep and smooth is not None:
                kp = side_keypoints(xy, "mid" if ex.use_midline else side)
                near_bottom = smooth < (ex.start + ex.depth) / 2
                for chk in ex.checks:
                    if chk.only_near_bottom and not near_bottom:
                        streak[chk.message] = 0
                        continue
                    try:
                        bad = chk.bad(kp)
                    except Exception:
                        bad = False
                    streak[chk.message] = streak[chk.message] + 1 if bad else 0
                    if streak[chk.message] >= BAD_STREAK:
                        counter.flag(chk.message)
                        live_issue = chk.message

            rep = counter.update(smooth, frame_i)
            if rep is not None:
                if rep.issues:
                    message, msg_good = rep.issues[0], False
                elif not rep.full:
                    message, msg_good = ex.shallow_msg, False
                else:
                    message, msg_good = ex.good_msg, True
                msg_left = MESSAGE_FRAMES
            elif live_issue:
                message, msg_good, msg_left = live_issue, False, 10

            render.draw_skeleton(frame, xy, conf, side, {a, b, c}, s, both_sides=ex.use_midline)
            if smooth is not None:
                render.draw_joint_angle(frame, anchor, smooth, s, unit=ex.unit)

        series.append((frame_i, raw, smooth))
        progress_val = None
        if smooth is not None:
            progress_val = (ex.up - smooth) / (ex.up - ex.depth)
        render.draw_hud(frame, ex.name, counter.count, counter.full_count, smooth, progress_val,
                        message if msg_left > 0 else "", msg_good, s)
        msg_left -= 1
        writer.write(frame)
        frame_i += 1
        if progress and total:
            progress(frame_i / total)

    cap.release()
    writer.release()
    _to_h264(tmp, out_video)
    elapsed = time.time() - t0

    reps = [r.__dict__ for r in counter.reps]
    issue_counts = {}
    for r in counter.reps:
        for i in r.issues:
            issue_counts[i] = issue_counts.get(i, 0) + 1
    mins = [r.min_angle for r in counter.reps]
    summary = {
        "exercise": ex.name,
        "video": os.path.basename(video_path),
        "frames": frame_i,
        "duration_s": round(frame_i / fps, 1),
        "processing_fps": round(frame_i / elapsed, 1) if elapsed else None,
        "reps": counter.count,
        "full_reps": counter.full_count,
        "partial_reps": counter.count - counter.full_count,
        "avg_bottom": round(float(np.mean(mins)), 1) if mins else None,
        "issues": issue_counts,
        "rep_details": reps,
        "thresholds": {"up": ex.up, "start": ex.start, "depth": ex.depth, "unit": ex.unit},
    }
    plot_path = os.path.join(out_dir, f"{stem}_{ex.key}_angles.png")
    _plot(series, counter.reps, ex, fps, plot_path)
    with open(os.path.join(out_dir, f"{stem}_{ex.key}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    summary["annotated_video"] = out_video
    summary["angle_plot"] = plot_path
    return summary


def _plot(series, reps, ex, fps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [f / fps for f, _, _ in series]
    raw = [np.nan if r is None else r for _, r, _ in series]
    sm = [np.nan if v is None else v for _, _, v in series]
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=130)
    for r in reps:
        ax.axvspan(r.start_frame / fps, r.end_frame / fps, color="#22c55e" if r.full else "#f59e0b", alpha=0.15, lw=0)
        ax.text((r.start_frame + r.end_frame) / 2 / fps, ex.up + 8, str(r.index), ha="center", fontsize=9, color="#333")
    ax.plot(t, raw, color="#c3cdfc", lw=1, label="raw")
    ax.plot(t, sm, color="#2347e8", lw=2, label="smoothed (EMA)")
    for y, lab, ls in [(ex.up, "top", "--"), (ex.start, "rep start", ":"), (ex.depth, "full depth", "--")]:
        ax.axhline(y, color="#888", ls=ls, lw=1)
        ax.text(t[-1] if t else 0, y + 1.5, lab, ha="right", fontsize=8, color="#666")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("shoulder height (% torso)" if ex.unit == "%" else f"{ex.name.lower()} joint angle (°)")
    lo = np.nanmin(sm) if np.isfinite(np.nanmin(sm)) else 30
    ax.set_ylim(min(30, lo - 10) if ex.unit != "%" else max(0, lo - 10), 195 if ex.unit != "%" else max(120, np.nanmax(sm) + 15))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"{ex.name}: {len(reps)} reps (green = full, orange = partial)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
