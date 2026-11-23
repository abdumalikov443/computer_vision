# AI Fitness Coach

Counts reps and checks exercise form from a normal phone video, using a
pretrained **YOLO11-pose** model. No custom training: the model finds 17 body
keypoints per frame, and the coach turns them into joint angles, a rep counter
and form feedback.

Supported: **squat**, **push-up**, **bicep curl** (filmed from the side).

## How it works

```
video frame ─► YOLO11-pose ─► 17 keypoints ─► signal ─► smoothing ─► rep counter ─► feedback
                              (largest person)   │          (EMA)      (hysteresis)     + form checks
```

| Exercise | Signal | Full rep when | Form check |
|---|---|---|---|
| Squat | knee angle (hip–knee–ankle) | knee ≤ 100° | torso lean > 50° at the bottom → "Keep your chest up" |
| Push-up | shoulder height above wrists, % of torso length | ≤ 42 % | shoulder–hip–ankle < 150° → "Keep your hips in line" |
| Bicep curl | elbow angle (shoulder–elbow–wrist) | elbow ≤ 65° | upper arm swings > 35° → "Keep your elbow still" |

- **Camera side:** the side of the body with higher keypoint confidence is used, smoothed over time so it does not flip.
- **Rep counter:** a two-state machine (`up` ↔ `down`). A rep starts below `start` and ends above `up`; the gap between them (hysteresis) stops jitter from creating extra reps. Reps shorter than 4 frames are ignored.
- **Push-up:** lying sideways, the model often swaps the left and right arm, which makes the elbow angle jump. The averaged shoulder height does not care which arm is which, so it is far more stable.
- **Form issues** must last 3 frames in a row before they are flagged on a rep.

## Results on test videos

| Video | Exercise | Real reps | Counted | Full / partial | Speed (CPU, 4 cores) |
|---|---|---|---|---|---|
| squats_test (640×360) | Squat | 7 | 7 | 7 / 0 | 15.8 fps |
| my_squat (1280×720) | Squat | 4 | 4 | 4 / 0 | 14.4 fps |
| push_up (640×360) | Push-up | 5 | 5 | 5 / 0 | 18.0 fps |
| bicep_press (1280×720) | Bicep curl | 4 | 4 | 4 / 0 | 16.2 fps |

The first push-up version used the elbow angle and counted 4 reps (1 full);
switching to shoulder height fixed it (5/5).

## Run

```bash
pip install -r requirements.txt
python analyze.py my_video.mp4 --exercise squat --out outputs/   # annotated video, angle plot, JSON summary
python app.py                                                     # Gradio demo on http://localhost:7860
python -m pytest -q tests                                         # rep counter tests
```

The `yolo11n-pose.pt` weights download automatically on first run.

## Files

```
coach/geometry.py   joint angles, lean from vertical, EMA smoothing
coach/exercises.py  exercise definitions: signal, thresholds, form checks
coach/counter.py    rep counter state machine
coach/pipeline.py   video loop, person selection, outputs (video, plot, JSON)
coach/render.py     skeleton, angle label, HUD and feedback banner
app.py              Gradio demo
analyze.py          command line
tests/              counter and geometry tests
```

## Limits

- Needs a side view with the whole body in frame; front views make angles unreliable.
- Thresholds are hand-tuned on a few videos; more videos would make them better.
- One person per video (the largest one is tracked).
