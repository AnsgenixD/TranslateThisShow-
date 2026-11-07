#!/usr/bin/env python3
"""
shot_detector.py - Scene / Shot Boundary Detection for Video Diarization.

Detects hard cuts in video using HSV color histogram correlation across frames.
Falls back to scenedetect if installed, otherwise uses robust native OpenCV.
Outputs: output/diar_v2/shots.json
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np


def detect_shots_opencv(
    video_path: str,
    threshold: float = 0.45,
    min_shot_duration_sec: float = 0.5
) -> List[Dict[str, Any]]:
    print(f"🎬 Detecting shot boundaries on {video_path} via OpenCV HSV correlation...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps
    print(f"📹 Video: {total_frames} frames, {fps:.2f} fps, {total_sec:.1f}s")

    shots = []
    prev_hist = None
    shot_start_f = 0
    shot_start_t = 0.0

    min_shot_frames = int(min_shot_duration_sec * fps)

    # Sample every 2nd frame for 2x speedup
    step = 2
    f_idx = 0

    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if f_idx % step == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            if prev_hist is not None:
                # Compare histograms (Correlation: 1.0 = identical, lower = scene change)
                sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if sim < threshold and (f_idx - shot_start_f) >= min_shot_frames:
                    shot_end_f = f_idx
                    shot_end_t = round(shot_end_f / fps, 3)
                    shots.append({
                        "shot_id": len(shots) + 1,
                        "start_frame": shot_start_f,
                        "end_frame": shot_end_f,
                        "start_sec": shot_start_t,
                        "end_sec": shot_end_t,
                        "duration": round(shot_end_t - shot_start_t, 3)
                    })
                    shot_start_f = f_idx
                    shot_start_t = shot_end_t

            prev_hist = hist

        f_idx += 1

    cap.release()

    # Final shot
    if f_idx > shot_start_f:
        end_t = round(f_idx / fps, 3)
        shots.append({
            "shot_id": len(shots) + 1,
            "start_frame": shot_start_f,
            "end_frame": f_idx,
            "start_sec": shot_start_t,
            "end_sec": end_t,
            "duration": round(end_t - shot_start_t, 3)
        })

    elapsed = time.time() - t0
    print(f"✅ Discovered {len(shots)} camera shots in {elapsed:.1f}s")
    return shots


def get_or_detect_shots(video_path: str, output_json: str = "output/diar_v2/shots.json") -> List[Dict[str, Any]]:
    out_p = Path(output_json)
    if out_p.exists():
        print(f"📖 Loaded cached shots from {output_json}")
        with open(out_p, "r", encoding="utf-8") as f:
            return json.load(f)

    shots = detect_shots_opencv(video_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(shots, f, indent=2)
    print(f"💾 Saved shots to {output_json}")
    return shots


if __name__ == "__main__":
    vid = "output/ep1_lower_30m.mp4"
    if len(sys.argv) > 1:
        vid = sys.argv[1]
    get_or_detect_shots(vid)
