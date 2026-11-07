#!/usr/bin/env python3
"""
phase3_face_tracker_asd.py - Shot-Aware Face Tracking & Active Speaker Detection (ASD).

1. Reads camera shot boundaries from output/diar_v2/shots.json.
2. Detects & tracks faces across frames within each shot using InsightFace ArcFace + 68pt 3D mesh.
3. Computes geometric Lip Aperture Ratio (LAR) for each frame.
4. Identifies tracks by cosine similarity to the 12 confirmed actor roster profiles.
5. Saves face track manifest to output/diar_v2/face_tracks.json.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from insightface.app import FaceAnalysis


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def compute_lar_68(landmarks_68: np.ndarray) -> float:
    """Calculates Lip Aperture Ratio: vertical inner lip distance / horizontal mouth width."""
    if len(landmarks_68) < 68:
        return 0.0
    corner_l = landmarks_68[48][:2]
    corner_r = landmarks_68[54][:2]
    lip_top = landmarks_68[62][:2]
    lip_bot = landmarks_68[66][:2]
    mouth_w = float(np.linalg.norm(corner_l - corner_r))
    inner_h = float(np.linalg.norm(lip_top - lip_bot))
    if mouth_w < 5.0:
        return 0.0
    return float(inner_h / (mouth_w + 1e-6))


def run_face_tracking_and_asd(
    video_path: str = "output/ep1_lower_30m.mp4",
    shots_json: str = "output/diar_v2/shots.json",
    roster_json: str = "output/face_roster/confirmed_actor_roster.json",
    output_json: str = "output/diar_v2/face_tracks.json",
    frame_step: int = 3,  # Sample every 3rd frame (~8.3 fps) for fast processing
):
    print("=" * 70)
    print("👁️ PHASE 3: SHOT-AWARE FACE TRACKING & ACTIVE SPEAKER DETECTION")
    print("=" * 70)

    t_start = time.time()

    # Load confirmed actor embeddings
    with open(roster_json, "r", encoding="utf-8") as f:
        roster_data = json.load(f)["confirmed_actors"]

    actor_embs = {}
    for k, v in roster_data.items():
        emb = np.array(v["embedding"], dtype=np.float32)
        actor_embs[k] = emb / (np.linalg.norm(emb) + 1e-6)

    print(f"👥 Loaded {len(actor_embs)} confirmed actor facial templates.")

    # Load shots
    with open(shots_json, "r", encoding="utf-8") as f:
        shots = json.load(f)
    print(f"🎬 Loaded {len(shots)} camera shots.")

    # Init InsightFace
    print("🧠 Initializing InsightFace ArcFace + 68pt 3D Landmark Mesh...")
    app = FaceAnalysis(
        name="buffalo_s",
        allowed_modules=["detection", "recognition", "landmark_3d_68"],
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    all_tracks: List[Dict[str, Any]] = []
    track_counter = 1

    total_shots = len(shots)
    print(f"🔍 Tracking faces across {total_shots} shots...")

    for shot_idx, shot in enumerate(shots):
        if (shot_idx + 1) % 50 == 0 or shot_idx == 0:
            print(f"  -> Shot {shot_idx + 1}/{total_shots} (at {shot['start_sec']}s, {len(all_tracks)} tracks found)...")

        s_frame = shot["start_frame"]
        e_frame = shot["end_frame"]

        if (e_frame - s_frame) < 5:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
        active_shot_tracks = []  # list of track objects for current shot

        for f_no in range(s_frame, e_frame):
            if (f_no - s_frame) % frame_step != 0:
                cap.grab()
                continue
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t_sec = round(f_no / fps, 3)
            faces = app.get(frame)

            # Filter valid faces
            valid_faces = []
            for face in faces:
                if face.det_score < 0.50:
                    continue
                bbox = face.bbox
                h_box = bbox[3] - bbox[1]
                w_box = bbox[2] - bbox[0]
                if h_box < 24 or w_box < 24:
                    continue
                valid_faces.append(face)

            # Match valid faces to active tracks using IoU
            unmatched_faces = list(range(len(valid_faces)))
            if active_shot_tracks:
                for trk in active_shot_tracks:
                    last_box = trk["bboxes"][-1]
                    best_iou = 0.0
                    best_f_idx = -1
                    for vf_i in unmatched_faces:
                        iou = compute_iou(last_box, valid_faces[vf_i].bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_f_idx = vf_i

                    if best_iou >= 0.35 and best_f_idx != -1:
                        f_match = valid_faces[best_f_idx]
                        trk["frames"].append(f_no)
                        trk["timestamps"].append(t_sec)
                        trk["bboxes"].append(f_match.bbox.tolist())
                        trk["embeddings"].append(f_match.normed_embedding)
                        lar = compute_lar_68(f_match.landmark_3d_68) if hasattr(f_match, "landmark_3d_68") and f_match.landmark_3d_68 is not None else 0.0
                        trk["lar_series"].append(lar)
                        unmatched_faces.remove(best_f_idx)

            # Create new tracks for remaining faces
            for vf_i in unmatched_faces:
                f_new = valid_faces[vf_i]
                lar = compute_lar_68(f_new.landmark_3d_68) if hasattr(f_new, "landmark_3d_68") and f_new.landmark_3d_68 is not None else 0.0
                active_shot_tracks.append({
                    "shot_id": shot["shot_id"],
                    "frames": [f_no],
                    "timestamps": [t_sec],
                    "bboxes": [f_new.bbox.tolist()],
                    "embeddings": [f_new.normed_embedding],
                    "lar_series": [lar]
                })

        # Finalize tracks for this shot
        for trk in active_shot_tracks:
            n_pts = len(trk["timestamps"])
            dur = trk["timestamps"][-1] - trk["timestamps"][0]
            if n_pts < 2 or dur < 0.3:
                continue

            # Average face embedding
            mean_emb = np.mean(trk["embeddings"], axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-6)

            # Identify actor
            sims = {act_id: float(np.dot(mean_emb, emb)) for act_id, emb in actor_embs.items()}
            sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
            best_actor, best_sim = sorted_sims[0]
            second_actor, second_sim = sorted_sims[1]

            if best_sim >= 0.58 and (best_sim - second_sim) >= 0.08:
                assigned_actor = best_actor
            else:
                assigned_actor = "UNKNOWN_FACE"

            # Compute Active Speaking Detection metrics
            lar_arr = np.array(trk["lar_series"])
            lar_std = float(np.std(lar_arr))
            lar_max = float(np.max(lar_arr))
            lar_mean = float(np.mean(lar_arr))

            # A speaking face has dynamic mouth aperture changes (standard deviation > 0.025 and max LAR > 0.12)
            is_speaking = (lar_std >= 0.025 and lar_max >= 0.12) or (lar_mean >= 0.15)
            asd_score = float(np.clip((lar_std * 15.0 + lar_mean * 2.0), 0.0, 1.0))

            all_tracks.append({
                "track_id": track_counter,
                "shot_id": trk["shot_id"],
                "actor_id": assigned_actor,
                "identity_sim": round(best_sim, 3),
                "start_sec": trk["timestamps"][0],
                "end_sec": trk["timestamps"][-1],
                "duration": round(dur, 3),
                "num_frames": n_pts,
                "is_speaking": bool(is_speaking),
                "asd_score": round(asd_score, 3),
                "lar_mean": round(lar_mean, 3),
                "lar_std": round(lar_std, 3)
            })
            track_counter += 1

    cap.release()

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tracks": len(all_tracks),
            "tracks": all_tracks
        }, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n✅ Phase 3 Complete: Extracted {len(all_tracks)} face tracks in {elapsed:.1f}s")
    print(f"💾 Saved to {output_json}")
    return all_tracks


if __name__ == "__main__":
    run_face_tracking_and_asd()
