"""AI Fitness Coach: rep counting and form feedback on top of YOLO11-pose."""
from .exercises import EXERCISES, get_exercise
from .counter import RepCounter
from .pipeline import analyze_video

__all__ = ["EXERCISES", "get_exercise", "RepCounter", "analyze_video"]
