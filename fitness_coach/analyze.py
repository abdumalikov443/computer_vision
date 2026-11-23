"""CLI: python analyze.py VIDEO --exercise squat --out outputs/"""
import argparse
import json

from coach import analyze_video, EXERCISES

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI Fitness Coach: count reps and check form in a video")
    ap.add_argument("video")
    ap.add_argument("--exercise", "-e", choices=list(EXERCISES), required=True)
    ap.add_argument("--out", "-o", default="outputs")
    ap.add_argument("--weights", default="yolo11n-pose.pt")
    args = ap.parse_args()
    summary = analyze_video(args.video, args.exercise, args.out, weights=args.weights)
    summary.pop("rep_details")
    print(json.dumps(summary, indent=2))
