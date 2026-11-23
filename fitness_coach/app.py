"""Gradio demo: upload a side-view workout video, get reps, form feedback, pose snapshots and an annotated video."""
import os
import tempfile

import gradio as gr

from coach import EXERCISES, analyze_video

CHOICES = {ex.name: key for key, ex in EXERCISES.items()}
GRADIO_MAJOR = int(gr.__version__.split(".")[0])

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#hero {padding: 28px 30px; border-radius: 18px; color: #fff;
  background: radial-gradient(120% 140% at 0% 0%, #3b5bff 0%, #1f2fb8 45%, #0f1640 100%);
  box-shadow: 0 18px 40px -22px rgba(35,71,232,.8); margin-bottom: 6px;}
#hero h1 {margin: 0; font-size: 34px; letter-spacing: -.02em; color: #fff;}
#hero p {margin: 8px 0 14px; opacity: .88; font-size: 16px; max-width: 760px; color: #fff;}
#hero .pills span {display: inline-block; margin: 0 8px 6px 0; padding: 5px 12px; border-radius: 999px;
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); font-size: 13px;}
.panel {border-radius: 16px !important;}
#tips ul {margin: 4px 0 0 18px; padding: 0;} #tips li {margin: 3px 0; font-size: 14px; opacity: .85;}
#analyze {font-size: 17px !important; padding: 14px !important; border-radius: 12px !important;}
.stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;}
.stat {border-radius: 14px; padding: 16px 18px; border: 1px solid rgba(127,127,127,.25);
  background: rgba(127,127,127,.06);}
.stat .v {font-size: 34px; font-weight: 800; letter-spacing: -.03em; line-height: 1.05;}
.stat .l {font-size: 12px; text-transform: uppercase; letter-spacing: .1em; opacity: .7; margin-top: 4px;}
.stat.main {background: linear-gradient(135deg, #2347e8, #5b3cf0); color: #fff; border: none;}
.stat.main .l {opacity: .85;}
.stat.good .v {color: #16a34a;} .stat.warn .v {color: #ea7a0c;}
.feedback {margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center;}
.chip {padding: 7px 13px; border-radius: 999px; font-size: 14px; font-weight: 600;}
.chip.ok {background: rgba(22,163,74,.14); color: #16a34a;}
.chip.bad {background: rgba(234,122,12,.15); color: #ea7a0c;}
.meta {font-size: 13px; opacity: .7; margin-top: 10px;}
.empty {border: 2px dashed rgba(127,127,127,.3); border-radius: 16px; padding: 46px 24px; text-align: center;}
.empty b {font-size: 18px;} .empty p {opacity: .7; margin: 6px 0 0;}
"""

HERO = """
<div id="hero">
  <h1>AI Fitness Coach</h1>
  <p>Upload a workout video. Pose estimation finds 17 body keypoints in every frame, then the coach
  counts your reps, measures how deep each one goes and tells you what to fix.</p>
  <div class="pills"><span>YOLO11-pose</span><span>17 keypoints / frame</span>
  <span>Squat · Push-up · Bicep curl</span><span>Runs on CPU</span></div>
</div>
"""

EMPTY = """<div class="empty"><b>Your results will appear here</b>
<p>Choose a video and an exercise, then press <b>Analyze</b>.</p></div>"""


def stats_html(s):
    unit = s["thresholds"]["unit"]
    avg = "—" if s["avg_bottom"] is None else f"{s['avg_bottom']:.0f}{unit}"
    chips = []
    if s["reps"] == 0:
        chips.append('<span class="chip bad">No reps found — film from the side, whole body in frame</span>')
    elif s["issues"]:
        chips += [f'<span class="chip bad">{m} · {n} rep(s)</span>' for m, n in s["issues"].items()]
    if s["partial_reps"]:
        chips.append(f'<span class="chip bad">{s["partial_reps"]} partial rep(s) — go through the full range</span>')
    if s["reps"] and not s["issues"] and not s["partial_reps"]:
        chips.append('<span class="chip ok">Clean set — every rep was full, no form issues</span>')
    return f"""
<div class="stats">
  <div class="stat main"><div class="v">{s['reps']}</div><div class="l">{s['exercise']} reps</div></div>
  <div class="stat good"><div class="v">{s['full_reps']}</div><div class="l">full reps</div></div>
  <div class="stat {'warn' if s['partial_reps'] else ''}"><div class="v">{s['partial_reps']}</div><div class="l">partial reps</div></div>
  <div class="stat"><div class="v">{avg}</div><div class="l">avg lowest point</div></div>
</div>
<div class="feedback">{''.join(chips)}</div>
<div class="meta">{s['duration_s']} s video · {s['frames']} frames analysed at {s['processing_fps']} fps ·
full rep needs ≤ {s['thresholds']['depth']}{unit}</div>"""


def run(video, exercise_name, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Upload a video first.")
    out_dir = tempfile.mkdtemp(prefix="coach_")
    s = analyze_video(video, CHOICES[exercise_name], out_dir,
                      progress=lambda p: progress(p, desc="Running pose estimation"))
    rows = [[r["index"], f"{r['min_angle']}{s['thresholds']['unit']}",
             "✅ full" if r["full"] else "⚠️ partial", ", ".join(r["issues"]) or "—"]
            for r in s["rep_details"]]
    return stats_html(s), s["annotated_video"], s["snapshots"], s["angle_plot"], rows


blocks_kwargs = {"title": "AI Fitness Coach"}
theme = gr.themes.Soft(primary_hue="blue", radius_size="lg", font=[gr.themes.GoogleFont("Inter"), "sans-serif"])
if GRADIO_MAJOR < 6:
    blocks_kwargs.update(css=CSS, theme=theme)

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(HERO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            with gr.Group(elem_classes="panel"):
                video = gr.Video(label="1 · Workout video", height=300)
                exercise = gr.Radio(list(CHOICES), value="Squat", label="2 · Exercise")
            btn = gr.Button("Analyze", variant="primary", elem_id="analyze")
            gr.HTML("""<div id="tips"><b>Tips for a good result</b><ul>
              <li>Film from the <b>side</b>, whole body in frame</li>
              <li>One person, steady camera, good light</li>
              <li>3–10 reps, 10–30 seconds is enough</li></ul></div>""")
        with gr.Column(scale=7):
            stats = gr.HTML(EMPTY)
            with gr.Tabs():
                with gr.Tab("Coach video"):
                    out_video = gr.Video(label="Coach video", show_label=False, height=420)
                with gr.Tab("Pose snapshots"):
                    gallery = gr.Gallery(label="Pose at the lowest point of every rep", show_label=False, columns=3, height=420, object_fit="contain")
                with gr.Tab("Movement chart"):
                    plot = gr.Image(label="Joint angle over time (green = full rep, orange = partial)")
                with gr.Tab("Rep details"):
                    table = gr.Dataframe(headers=["rep", "lowest point", "result", "form notes"], interactive=False)
    btn.click(run, [video, exercise], [stats, out_video, gallery, plot, table])

if __name__ == "__main__":
    launch_kwargs = {"server_name": "0.0.0.0", "server_port": int(os.environ.get("PORT", 7860))}
    if GRADIO_MAJOR >= 6:
        launch_kwargs.update(css=CSS, theme=theme)
    demo.launch(**launch_kwargs)
