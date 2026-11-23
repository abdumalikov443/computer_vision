"""Gradio demo: upload a side-view workout video, get reps, feedback and an annotated video."""
import os
import tempfile

import gradio as gr

from coach import EXERCISES, analyze_video

CHOICES = {ex.name: key for key, ex in EXERCISES.items()}


def run(video, exercise_name, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload a video first.")
    out_dir = tempfile.mkdtemp(prefix="coach_")
    s = analyze_video(video, CHOICES[exercise_name], out_dir, progress=lambda p: progress(p, desc="Analyzing"))
    unit = s["thresholds"]["unit"]
    lines = [
        f"## {s['reps']} reps",
        f"**{s['full_reps']} full** · {s['partial_reps']} partial · {s['duration_s']} s video",
        "",
    ]
    if s["avg_bottom"] is not None:
        lines.append(f"Average lowest point: **{s['avg_bottom']}{unit}** (full rep needs ≤ {s['thresholds']['depth']}{unit})")
    if s["issues"]:
        lines.append("\n**Form notes**")
        lines += [f"- {msg}: {n} rep(s)" for msg, n in s["issues"].items()]
    else:
        lines.append("\nNo form issues detected.")
    rows = [[r["index"], r["min_angle"], "full" if r["full"] else "partial", ", ".join(r["issues"]) or "—"]
            for r in s["rep_details"]]
    return s["annotated_video"], "\n".join(lines), s["angle_plot"], rows


with gr.Blocks(title="AI Fitness Coach") as demo:
    gr.Markdown("# AI Fitness Coach\nRep counting and form feedback with **YOLO11-pose**. "
                "Film from the **side**, whole body in frame.")
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Workout video")
            exercise = gr.Radio(list(CHOICES), value="Squat", label="Exercise")
            btn = gr.Button("Analyze", variant="primary")
            summary = gr.Markdown()
        with gr.Column(scale=1):
            out_video = gr.Video(label="Coach view")
            plot = gr.Image(label="Movement over time")
    table = gr.Dataframe(headers=["rep", "lowest point", "result", "form notes"], label="Reps")
    btn.click(run, [video, exercise], [out_video, summary, plot, table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
