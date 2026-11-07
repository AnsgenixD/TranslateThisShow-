"""
Stage 2: Video-Guided Active Speaker Reference Puller & Interactive Review Studio.

Integrated with Stage 1 (ArcFace 512-D Face Roster):
1. Loads confirmed actor identities & 512-D ArcFace embeddings from Stage 1 (`confirmed_actor_roster.json`).
2. Scans subtitle dialogue segments, detects on-screen faces via InsightFace, and matches them to confirmed actors.
3. Tracks mouth ROI dynamics across dialogue frames to confirm the person is actively speaking (eliminating silent reaction shots).
4. Slices BOTH synchronized .mp4 video snippet previews and clean 24kHz normalized .wav vocal stems (resampled with scipy).
5. Generates the interactive Drag-and-Drop HTML5 Review Studio:
   - Inline video + audio playback of the exact dialogue moment.
   - Actor portraits, names, and Stage 1 cumulative screen time.
   - Automatic reserve promotion when clips are dropped or deleted.
   - One-click export of confirmed actor reference audio profiles for TTS dubbing!
"""
from __future__ import annotations

import os
import re
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from insightface.app import FaceAnalysis

try:
    from CleanProject.src.gender_pitch_gate import GenderPitchGate
    from CleanProject.src.lip_sync_asd import LipSyncASD
except ImportError:
    from src.gender_pitch_gate import GenderPitchGate
    from src.lip_sync_asd import LipSyncASD

try:
    from CleanProject.src.face_roster_discovery import discover_face_roster
except ImportError:
    from src.face_roster_discovery import discover_face_roster


def parse_srt(path: str) -> List[dict]:
    """Parse SRT subtitle file into structured list of dialogue lines."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\d+\n|\Z)"
    )
    entries = []
    for m in pattern.finditer(text):
        idx, start_str, end_str, content = m.groups()
        h, mn, s_ms = start_str.split(":")
        s, ms = s_ms.split(",")
        start_sec = int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        h, mn, s_ms = end_str.split(":")
        s, ms = s_ms.split(",")
        end_sec = int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        entries.append({
            "index": int(idx),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(end_sec - start_sec, 3),
            "start_str": start_str,
            "end_str": end_str,
            "text": content.strip().replace("\n", " "),
        })
    return entries


def get_mouth_roi_gray(frame: np.ndarray, f_kps: np.ndarray) -> Optional[np.ndarray]:
    """Extract a normalized 40x40 grayscale mouth region using InsightFace 5-point landmarks."""
    r_mouth, l_mouth = f_kps[3], f_kps[4]
    mouth_w = float(np.linalg.norm(r_mouth - l_mouth))
    if mouth_w < 8:
        return None
    mouth_center = (r_mouth + l_mouth) / 2.0
    r = int(mouth_w * 0.75)
    h_f, w_f = frame.shape[:2]
    x1, x2 = max(0, int(mouth_center[0] - r)), min(w_f, int(mouth_center[0] + r))
    y1, y2 = max(0, int(mouth_center[1] - r)), min(h_f, int(mouth_center[1] + r))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (40, 40))


def slice_video_snippet(video_path: str, start_s: float, end_s: float, out_mp4: Path):
    """Slices a lightweight mp4 video snippet using ffmpeg stream copy."""
    pad_start = max(0.0, start_s - 0.2)
    pad_end = end_s + 0.2
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(pad_start),
        "-to", str(pad_end),
        "-i", str(video_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_mp4)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def run_visual_actor_extraction(
    video_path: str = "output/ep1_lower_30m.mp4",
    vocals_path: str = "output/ep1_lower_30m_stems/htdemucs/ep1_lower_30m_24k/vocals.wav",
    srt_path: str = "output/ep1_lower_30m_chinese.srt",
    roster_path: str = "output/face_roster/confirmed_actor_roster.json",
    output_dir: str = "output/visual_actors",
    similarity_threshold: float = 0.52,
    target_audio_sr: int = 24000,
    clips_per_actor: int = 7,  # 4 active + 3 reserve
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🎬 STAGE 2: VIDEO-GUIDED ACTOR REFERENCE PULLER (ArcFace 512-D)")
    print(f"📹 Video Stream  : {video_path}")
    print(f"🎙️ Vocal Stem    : {vocals_path}")
    print(f"📄 Subtitles     : {srt_path}")
    print(f"👥 Actor Roster  : {roster_path}")
    print(f"📂 Output Folder : {out_dir.resolve()}")
    print("=" * 70)

    # 1. Load Confirmed Actor Roster from Stage 1
    r_path = Path(roster_path)
    if not r_path.exists():
        manifest_fallback = r_path.parent / "roster_manifest.json"
        if manifest_fallback.exists():
            print(f"ℹ️ {roster_path} not found. Loading from roster manifest {manifest_fallback}...")
            r_path = manifest_fallback
        else:
            print("⚠️ No face roster found. Automatically running Stage 1 Face Discovery first...")
            discover_face_roster(video_path=video_path, output_dir=str(r_path.parent))

    roster_data = json.loads(r_path.read_text(encoding="utf-8"))
    if "confirmed_actors" in roster_data and roster_data["confirmed_actors"]:
        confirmed_actors = roster_data["confirmed_actors"]
    elif "clusters" in roster_data:
        confirmed_actors = {
            k: v for k, v in roster_data["clusters"].items()
            if v.get("is_main_cast_suggested", True)
        }
    else:
        raise RuntimeError(f"Could not load actor identities from {roster_path}")

    print(f"👥 Loaded {len(confirmed_actors)} confirmed actor identities from Stage 1.")
    actor_embs = {
        k: np.array(v["embedding"]) / (np.linalg.norm(v["embedding"]) + 1e-6)
        for k, v in confirmed_actors.items()
    }

    # 2. Initialize InsightFace RetinaFace + ArcFace + GenderAge
    print("\n🧠 Loading InsightFace ArcFace & Demographic models...")
    app = FaceAnalysis(
        name="buffalo_s",
        root=str(Path.home() / ".insightface"),
        allowed_modules=["detection", "recognition", "genderage"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    gender_gate = GenderPitchGate()
    lip_asd = LipSyncASD()
    actor_genders: Dict[str, str] = {}
    roster_dir = Path(roster_path).parent
    for ak in confirmed_actors.keys():
        crops = list((roster_dir / ak).glob("*.jpg"))
        v_res = gender_gate.determine_actor_visual_gender(app, crops)
        actor_genders[ak] = v_res["gender"]
        confirmed_actors[ak]["verified_gender"] = v_res["gender"]
    print(f"👥 Determined canonical visual genders: {actor_genders}")

    # 3. Read Subtitles and Audio Stem
    entries = parse_srt(srt_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    print(f"🎙️ Loading vocal stem audio ({vocals_path})...")
    audio_full, audio_sr = sf.read(vocals_path, dtype="float32")
    if audio_full.ndim > 1:
        audio_mono = np.mean(audio_full, axis=1)
    else:
        audio_mono = audio_full

    verified_actor_shots: Dict[str, List[Dict[str, Any]]] = {k: [] for k in confirmed_actors.keys()}
    total_scanned = 0
    t0 = time.time()

    print(f"\n🔍 Scanning {len(entries)} dialogue lines for verified actor speech...")

    for idx, e in enumerate(entries):
        start = e["start"]
        end = e["end"]
        dur = end - start
        if dur < 0.6:
            continue

        n_samples = 5 if dur >= 1.5 else (4 if dur >= 0.9 else 3)
        ts_list = np.linspace(start + 0.05, end - 0.05, n_samples)
        frame_detections = []

        for ts in ts_list:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            faces = app.get(frame)
            valid = [
                f for f in faces
                if f.det_score >= 0.55 and ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) >= 1200
            ]
            if valid:
                frame_detections.append((frame, valid))

        if len(frame_detections) < 2:
            continue

        # Match faces in frames against confirmed actors
        actor_frame_instances: Dict[str, List[Tuple[float, np.ndarray, Any, np.ndarray]]] = {}
        total_faces_per_frame = [len(faces) for _, faces in frame_detections]

        for frame, faces in frame_detections:
            for f in faces:
                sims = {k: float(np.dot(f.normed_embedding, emb)) for k, emb in actor_embs.items()}
                best_k = max(sims, key=sims.get)
                if sims[best_k] >= similarity_threshold:
                    m_gray = get_mouth_roi_gray(frame, f.kps)
                    aperture = lip_asd.get_lip_aperture(frame=frame, face_dict=f) or 0.0
                    actor_frame_instances.setdefault(best_k, []).append((sims[best_k], m_gray, f, frame, aperture))

        # Check audio chunk for active speech correlation and pitch
        s_samp = int(start * audio_sr)
        e_samp = int(end * audio_sr)
        chunk = audio_mono[s_samp:e_samp]

        # Check for active speaker among matching confirmed actors
        for ak, inst_list in actor_frame_instances.items():
            if len(inst_list) >= 2:
                # 1. Lip-Sync Cross-Modal Temporal Synchrony (SyncNet principle)
                apertures = [item[4] for item in inst_list]
                sync_score = lip_asd.compute_synchrony_score(apertures, chunk, audio_sr)
                if sync_score < 0.28:
                    continue  # Reaction shot: lips not in sync with audio

                # 2. Demographic & Pitch Gate: Audio F0 must match actor's visual gender
                cand_gender = actor_genders.get(ak, "unknown")
                if cand_gender != "unknown":
                    is_valid, reason, ac_info = gender_gate.validate_clip_pair(cand_gender, chunk, audio_sr)
                    if not is_valid:
                        continue  # Reaction shot: off-screen speaker of different gender

                # Calculate mouth motion diff
                diffs = []
                for i in range(1, len(inst_list)):
                    m1, m2 = inst_list[i - 1][1], inst_list[i][1]
                    if m1 is not None and m2 is not None:
                        diffs.append(float(np.mean(np.abs(m2.astype(float) - m1.astype(float)))))
                mouth_diff = float(np.mean(diffs)) if diffs else 0.0

                is_solo = max(total_faces_per_frame) == 1
                min_motion_req = 1.0 if is_solo else 1.8

                if mouth_diff >= min_motion_req:
                    sim_avg = float(np.mean([item[0] for item in inst_list]))
                    best_inst = max(inst_list, key=lambda x: x[0])
                    best_f = best_inst[2]
                    best_frame = best_inst[3]

                    # Contextual face crop
                    h, w = best_frame.shape[:2]
                    b = best_f.bbox
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    cx1 = max(0, int(b[0] - bw * 0.2))
                    cy1 = max(0, int(b[1] - bh * 0.3))
                    cx2 = min(w, int(b[2] + bw * 0.2))
                    cy2 = min(h, int(b[3] + bh * 0.2))
                    crop = best_frame[cy1:cy2, cx1:cx2]

                    verified_actor_shots[ak].append({
                        "entry": e,
                        "similarity": sim_avg,
                        "mouth_motion": mouth_diff,
                        "lip_sync": sync_score,
                        "is_solo": is_solo,
                        "crop": crop,
                    })
                    total_scanned += 1
                    break  # One active speaker per dialogue segment

        if (idx + 1) % 150 == 0 or (idx + 1) == len(entries):
            elapsed = time.time() - t0
            print(f"  -> Scanned {idx + 1}/{len(entries)} lines ({total_scanned} verified active dialogue clips, {elapsed:.1f}s)")

    cap.release()
    print(f"\n✅ Scan complete in {time.time() - t0:.1f}s! Found {total_scanned} confirmed dialogue moments.")

    # 4. Slicing Audio-Video Clips for Each Confirmed Actor
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_path": video_path,
        "vocals_path": vocals_path,
        "srt_path": srt_path,
        "total_dialogue_lines": len(entries),
        "total_verified_clips": total_scanned,
        "actors": {},
    }

    print(f"\n📦 Slicing video snippet previews and 24kHz vocal stems for {len(confirmed_actors)} actors...")

    for rank_idx, (ak, a_info) in enumerate(confirmed_actors.items(), 1):
        actor_shots = verified_actor_shots.get(ak, [])
        actor_folder = out_dir / ak
        actor_folder.mkdir(parents=True, exist_ok=True)

        actor_name = a_info.get("name", ak)
        screen_time_fmt = a_info.get("screen_time_formatted", "0m 00s")

        # Copy Stage 1 Avatar
        avatar_path = actor_folder / "avatar.jpg"
        source_avatar = Path(roster_path).parent / a_info.get("primary_avatar", "")
        if source_avatar.exists():
            shutil.copy2(source_avatar, avatar_path)
        elif actor_shots and actor_shots[0]["crop"].size > 0:
            cv2.imwrite(str(avatar_path), actor_shots[0]["crop"])

        # Score & Rank Speech Clips
        def _clip_score(s):
            dur = float(s["entry"]["duration"])
            dur_bonus = 1.0 if (2.2 <= dur <= 5.5) else (0.7 if (1.5 <= dur <= 8.0) else 0.4)
            solo_bonus = 1.4 if s["is_solo"] else 1.0
            return dur_bonus * solo_bonus * s["similarity"] * min(2.5, s["mouth_motion"])

        sorted_shots = sorted(actor_shots, key=_clip_score, reverse=True)
        chosen_shots = sorted_shots[:clips_per_actor]

        # Calculate Confidence
        solo_shots = sum(1 for s in actor_shots if s["is_solo"])
        volume_pts = 40 if len(actor_shots) >= 15 else (28 if len(actor_shots) >= 6 else (16 if len(actor_shots) >= 2 else 6))
        solo_pts = int((solo_shots / max(1, len(actor_shots))) * 35)
        sim_pts = int(min(25, (np.mean([s["similarity"] for s in actor_shots]) if actor_shots else 0.5) * 35))
        confidence_pct = min(99, max(25, volume_pts + solo_pts + sim_pts))

        if confidence_pct >= 80:
            tier_label = "High Confidence"
            tier_color = "#3fb950"
        elif confidence_pct >= 60:
            tier_label = "Medium Confidence"
            tier_color = "#d29922"
        else:
            tier_label = "Low Confidence (Review)"
            tier_color = "#f85149"

        samples_meta = []
        for s_rank, shot in enumerate(chosen_shots, 1):
            e = shot["entry"]
            start_s = float(e["start"])
            end_s = float(e["end"])

            # A. Sliced 24kHz Normalized WAV
            s_samp = int(start_s * audio_sr)
            e_samp = int(end_s * audio_sr)
            clip = audio_mono[s_samp:e_samp]
            if len(clip) == 0:
                continue

            peak = float(np.max(np.abs(clip))) + 1e-6
            norm_clip = clip / peak * 0.90

            wav_name = f"sample_{s_rank:02d}.wav"
            txt_name = f"sample_{s_rank:02d}.txt"
            mp4_name = f"sample_{s_rank:02d}.mp4"

            wav_path = actor_folder / wav_name
            txt_path = actor_folder / txt_name
            mp4_path = actor_folder / mp4_name

            # Fast polyphase resample if needed
            if audio_sr != target_audio_sr:
                if audio_sr == 44100 and target_audio_sr == 24000:
                    clip_out = resample_poly(norm_clip, 80, 147)
                elif audio_sr == 48000 and target_audio_sr == 24000:
                    clip_out = resample_poly(norm_clip, 1, 2)
                else:
                    clip_out = norm_clip
                sf.write(str(wav_path), clip_out, target_audio_sr)
            else:
                sf.write(str(wav_path), norm_clip, target_audio_sr)

            # B. Sliced MP4 video snippet
            slice_video_snippet(video_path, start_s, end_s, mp4_path)

            txt_content = (
                f"Actor: {actor_name} ({ak})\n"
                f"Line #{int(e['index'])} [{e['start_str']} --> {e['end_str']}] (Duration: {float(e['duration']):.2f}s, Sim: {shot['similarity']:.2f}, Motion: {shot['mouth_motion']:.2f})\n"
                f"{e['text']}\n"
            )
            txt_path.write_text(txt_content, encoding="utf-8")

            samples_meta.append({
                "clip_id": f"{ak}_c{s_rank}",
                "rank": int(s_rank),
                "line_index": int(e["index"]),
                "start": start_s,
                "end": end_s,
                "duration": float(e["duration"]),
                "text": str(e["text"]),
                "motion": round(float(shot["mouth_motion"]), 2),
                "similarity": round(float(shot["similarity"]), 2),
                "is_solo": bool(shot["is_solo"]),
                "wav_path": str(wav_path.relative_to(out_dir)),
                "mp4_path": str(mp4_path.relative_to(out_dir)),
                "is_reserve": s_rank > 4,  # First 4 displayed in active grid, rest reserve
            })

        total_speech_sec = sum(float(s["entry"]["duration"]) for s in actor_shots)
        actor_data = {
            "actor_id": int(rank_idx),
            "label": ak,
            "name": actor_name,
            "screen_time_formatted": screen_time_fmt,
            "confidence_pct": confidence_pct,
            "confidence_tier": tier_label,
            "tier_color": tier_color,
            "avatar_image": str(avatar_path.relative_to(out_dir)),
            "total_shots_seen": int(len(actor_shots)),
            "solo_shots": int(solo_shots),
            "total_speech_sec": round(float(total_speech_sec), 2),
            "samples": samples_meta,
        }

        (actor_folder / "summary.json").write_text(json.dumps(actor_data, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["actors"][ak] = actor_data

        first_txt = samples_meta[0]["text"] if samples_meta else "(no solo speech detected)"
        print(f"  🎭 #{rank_idx:02d} {ak} ({actor_name}) [{confidence_pct}% {tier_label}] ({len(actor_shots)} lines, {total_speech_sec:.1f}s speech): {len(samples_meta)} clips ready -> '{first_txt}'")

    # Save manifest
    manifest_file = out_dir / "visual_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Autonomous Auto-Export of Confirmed Actor References JSON for Downstream TTS/Diarizer
    confirmed_ref_path = Path("output/confirmed_actor_references.json")
    confirmed_data = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "actors": {}
    }
    for ak, adata in manifest["actors"].items():
        active = [s for s in adata.get("samples", []) if not s.get("is_reserve")]
        confirmed_data["actors"][ak] = {
            "key": ak,
            "name": adata.get("name", ak),
            "screen_time_formatted": adata.get("screen_time_formatted", ""),
            "confidence_pct": adata.get("confidence_pct", 80),
            "avatar_image": adata.get("avatar_image", ""),
            "verified_gender": actor_genders.get(ak, "unknown"),
            "active_clips": active
        }
    confirmed_ref_path.write_text(json.dumps(confirmed_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📦 Auto-exported verified references: {confirmed_ref_path.resolve()}")

    # Generate interactive Drag-and-Drop Video Review HTML
    generate_interactive_video_html(manifest, out_dir / "visual_review_player.html")

    print(f"\n🎉 Stage 2 Actor Reference Pulling Complete!")
    print(f"📄 Manifest:      {manifest_file.resolve()}")
    print(f"🌐 Review Studio: {out_dir / 'visual_review_player.html'}")
    return manifest


def generate_interactive_video_html(manifest: dict, html_path: Path):
    actors = manifest.get("actors", {})
    actors_json = json.dumps(actors, ensure_ascii=False)

    actor_cards_html = []
    for actor_key, data in actors.items():
        actor_id = data["actor_id"]
        conf_pct = data["confidence_pct"]
        tier_label = data["confidence_tier"]
        tier_color = data["tier_color"]
        actor_name = data.get("name", actor_key)
        screen_time_fmt = data.get("screen_time_formatted", "")

        active_samples = [s for s in data.get("samples", []) if not s.get("is_reserve")]
        reserve_count = len([s for s in data.get("samples", []) if s.get("is_reserve")])

        clips_html = []
        for s in active_samples:
            clips_html.append(_render_clip_card(s, actor_id))

        actor_cards_html.append(f"""
        <div class="actor-card" id="actor-card-{actor_id}" data-actor-id="{actor_id}" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event, {actor_id})">
          <div class="avatar-col">
            <img class="avatar-img" src="{data['avatar_image']}" alt="{actor_key}">
            <div class="actor-tag">{actor_key}</div>
            <div class="conf-badge" style="background: {tier_color}22; color: {tier_color}; border: 1px solid {tier_color}66;">
              ★ {conf_pct}% {tier_label}
            </div>
            <div class="reserve-indicator" id="reserve-count-{actor_id}">
              {reserve_count} reserve clips ready
            </div>
          </div>
          <div class="info-col">
            <div class="card-top">
              <div>
                <h3 class="actor-title">{actor_name} <span class="actor-sub">({actor_key} • {screen_time_fmt} Screen Time)</span></h3>
                <p class="actor-stats">{data['total_shots_seen']} dialogue lines | {data['solo_shots']} solo shots | {data['total_speech_sec']}s active speech</p>
              </div>
              <div class="actor-actions">
                <input type="text" placeholder="Actor Name" id="name-input-{actor_id}" class="name-input" value="{actor_name}">
                <button class="confirm-btn" onclick="toggleConfirm({actor_id})">✅ Confirm</button>
                <button class="deny-btn" onclick="denyActor({actor_id})">🗑️ Discard</button>
              </div>
            </div>
            <div class="drop-hint">Drop dialogue clips here to reassign to {actor_name}</div>
            <div class="clips-grid" id="clips-grid-{actor_id}">
              {''.join(clips_html)}
            </div>
          </div>
        </div>
        """)

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stage 2: Video Actor Reference Studio & Voice Diarizer</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #090d13;
      color: #c9d1d9;
      margin: 0;
      padding: 0 0 100px 0;
    }}
    .sticky-bar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: #161b22f5;
      backdrop-filter: blur(14px);
      border-bottom: 1px solid #30363d;
      padding: 14px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .sticky-title h1 {{
      margin: 0;
      font-size: 20px;
      color: #58a6ff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .sticky-title p {{ margin: 4px 0 0 0; font-size: 13px; color: #8b949e; }}
    .bar-actions {{ display: flex; gap: 14px; align-items: center; }}
    .export-btn {{
      background: #238636;
      color: #fff;
      border: 1px solid rgba(240,246,252,0.1);
      padding: 9px 20px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(35,134,54,0.4);
      transition: background 0.15s, transform 0.1s;
    }}
    .export-btn:hover {{ background: #2ea043; transform: translateY(-1px); }}
    .container {{
      max-width: 1200px;
      margin: 24px auto;
      padding: 0 16px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    .actor-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 20px;
      display: flex;
      gap: 22px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.4);
      transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .actor-card.drag-target {{
      border-color: #58a6ff !important;
      box-shadow: 0 0 16px rgba(88,166,255,0.4);
    }}
    .actor-card.confirmed {{ border-left: 6px solid #238636; }}
    .actor-card.discarded {{ opacity: 0.4; filter: grayscale(80%); }}
    .avatar-col {{
      width: 140px;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }}
    .avatar-img {{
      width: 124px;
      height: 124px;
      border-radius: 10px;
      object-fit: cover;
      border: 2px solid #58a6ff;
      background: #0d1117;
    }}
    .actor-tag {{ font-weight: 600; font-size: 14px; color: #f0f6fc; }}
    .conf-badge {{
      font-size: 11px;
      padding: 3px 8px;
      border-radius: 12px;
      font-weight: 600;
      text-align: center;
    }}
    .reserve-indicator {{ font-size: 11px; color: #8b949e; text-align: center; }}
    .info-col {{
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid #21262d;
      padding-bottom: 10px;
    }}
    .actor-title {{ margin: 0 0 4px 0; font-size: 19px; color: #f0f6fc; }}
    .actor-sub {{ font-size: 13px; color: #8b949e; font-weight: 400; }}
    .actor-stats {{ margin: 0; font-size: 12px; color: #8b949e; }}
    .actor-actions {{ display: flex; gap: 8px; align-items: center; }}
    .name-input {{
      background: #0d1117;
      border: 1px solid #30363d;
      color: #fff;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      width: 180px;
    }}
    .confirm-btn, .deny-btn {{
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
    }}
    .confirm-btn {{ background: #23863622; color: #7ee787; border-color: #23863666; }}
    .confirm-btn:hover {{ background: #23863644; }}
    .deny-btn {{ background: #da363322; color: #f85149; border-color: #da363366; }}
    .deny-btn:hover {{ background: #da363344; }}
    .drop-hint {{
      font-size: 11px;
      color: #6e7681;
      text-align: center;
      padding: 4px;
      border: 1px dashed #30363d;
      border-radius: 6px;
      background: #0d111740;
    }}
    .clips-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }}
    @media (max-width: 1050px) {{
      .clips-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .clip-card {{
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      cursor: grab;
      transition: transform 0.1s, border-color 0.1s;
    }}
    .clip-card:hover {{ border-color: #58a6ff; transform: translateY(-2px); }}
    .clip-card.dragging {{ opacity: 0.4; }}
    .clip-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .clip-id {{ font-size: 11px; color: #7ee787; font-weight: 600; }}
    .delete-clip-btn {{
      background: transparent;
      border: none;
      color: #8b949e;
      cursor: pointer;
      font-size: 13px;
      padding: 0 4px;
    }}
    .delete-clip-btn:hover {{ color: #f85149; }}
    .clip-video {{
      width: 100%;
      height: 100px;
      background: #000;
      border-radius: 6px;
      outline: none;
    }}
    .clip-quote {{
      font-size: 12px;
      color: #c9d1d9;
      min-height: 34px;
      line-height: 1.35;
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }}
    .clip-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: auto;
      border-top: 1px solid #21262d;
      padding-top: 6px;
    }}
    .motion-tag {{ font-size: 10px; color: #8b949e; }}
    .toggle-audio-btn {{
      background: #21262d;
      border: 1px solid #30363d;
      color: #8b949e;
      padding: 3px 6px;
      border-radius: 4px;
      font-size: 10px;
      cursor: pointer;
    }}
    .toggle-audio-btn:hover {{ color: #fff; border-color: #58a6ff; }}
    .toast {{
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #1f6feb;
      color: #fff;
      padding: 10px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      box-shadow: 0 4px 14px rgba(0,0,0,0.5);
      display: none;
      z-index: 1000;
    }}
  </style>
</head>
<body>
  <div class="sticky-bar">
    <div class="sticky-title">
      <h1>🎬 Stage 2: Actor Reference Review Studio</h1>
      <p>Clean video previews & 24kHz vocal stems matched to Stage 1 confirmed actors. Drag clips between actors or delete to replace from reserve.</p>
    </div>
    <div class="bar-actions">
      <span class="toast" id="toast-msg">Action confirmed</span>
      <button class="export-btn" onclick="exportConfirmedActors()">🚀 Confirm & Export References</button>
    </div>
  </div>

  <div class="container">
    {''.join(actor_cards_html)}
  </div>

  <script>
    let actorsState = {actors_json};
    let draggedClipId = null;
    let draggedFromActorId = null;

    function handleDragStart(e, clipId, actorId) {{
      draggedClipId = clipId;
      draggedFromActorId = actorId;
      e.target.classList.add("dragging");
      e.dataTransfer.setData("text/plain", JSON.stringify({{ clipId, actorId }}));
    }}

    function handleDragEnd(e) {{
      e.target.classList.remove("dragging");
    }}

    function handleDragOver(e) {{
      e.preventDefault();
      const card = e.currentTarget;
      card.classList.add("drag-target");
    }}

    function handleDragLeave(e) {{
      const card = e.currentTarget;
      card.classList.remove("drag-target");
    }}

    function handleDrop(e, targetActorId) {{
      e.preventDefault();
      const card = e.currentTarget;
      card.classList.remove("drag-target");

      if (draggedFromActorId === targetActorId) return;

      const fromActorKey = Object.keys(actorsState).find(k => actorsState[k].actor_id === draggedFromActorId);
      const toActorKey = Object.keys(actorsState).find(k => actorsState[k].actor_id === targetActorId);

      const fromList = actorsState[fromActorKey].samples;
      const toList = actorsState[toActorKey].samples;

      const clipIdx = fromList.findIndex(s => s.clip_id === draggedClipId);
      if (clipIdx === -1) return;

      const [clipObj] = fromList.splice(clipIdx, 1);
      clipObj.is_reserve = false;
      toList.unshift(clipObj);

      // Promote reserve clip on donor actor
      const donorReserve = fromList.find(s => s.is_reserve);
      if (donorReserve) {{
        donorReserve.is_reserve = false;
      }}

      reRenderActor(draggedFromActorId);
      reRenderActor(targetActorId);
      showToast(`Moved clip to ${{actorsState[toActorKey].name}}!`);
    }}

    function deleteClip(clipId, actorId) {{
      const actorKey = Object.keys(actorsState).find(k => actorsState[k].actor_id === actorId);
      const list = actorsState[actorKey].samples;
      const idx = list.findIndex(s => s.clip_id === clipId);
      if (idx !== -1) {{
        list.splice(idx, 1);
        const nextReserve = list.find(s => s.is_reserve);
        if (nextReserve) {{
          nextReserve.is_reserve = false;
        }}
        reRenderActor(actorId);
        showToast("Clip removed and reserve promoted!");
      }}
    }}

    function reRenderActor(actorId) {{
      const actorKey = Object.keys(actorsState).find(k => actorsState[k].actor_id === actorId);
      const data = actorsState[actorKey];
      const grid = document.getElementById(`clips-grid-${{actorId}}`);
      const reserveTag = document.getElementById(`reserve-count-${{actorId}}`);

      const activeSamples = data.samples.filter(s => !s.is_reserve);
      const reserveCount = data.samples.filter(s => s.is_reserve).length;

      grid.innerHTML = activeSamples.map((s, idx) => {{
        s.rank = idx + 1;
        return `
        <div class="clip-card" id="card-${{s.clip_id}}" draggable="true" ondragstart="handleDragStart(event, '${{s.clip_id}}', ${{actorId}})" ondragend="handleDragEnd(event)">
          <div class="clip-header">
            <span class="clip-id">#${{s.rank}} (${{s.duration.toFixed(1)}}s)</span>
            <button class="delete-clip-btn" title="Delete clip" onclick="deleteClip('${{s.clip_id}}', ${{actorId}})">✕</button>
          </div>
          <video class="clip-video" controls preload="none" src="${{s.mp4_path}}"></video>
          <div class="clip-quote">"${{s.text}}"</div>
          <div class="clip-footer">
            <span class="motion-tag">Motion: ${{s.motion}} | Sim: ${{s.similarity}}</span>
            <button class="toggle-audio-btn" onclick="toggleIsolatedVocal(this, '${{s.wav_path}}')">🔊 Vocal Stem</button>
          </div>
        </div>
        `;
      }}).join("");

      if (reserveTag) {{
        reserveTag.textContent = `${{reserveCount}} reserve clips ready`;
      }}
    }}

    function toggleIsolatedVocal(btn, wavPath) {{
      const card = btn.closest(".clip-card");
      const video = card.querySelector(".clip-video");
      const isVocal = btn.dataset.mode === "vocal";

      if (isVocal) {{
        video.muted = false;
        if (card._audio) card._audio.pause();
        btn.dataset.mode = "original";
        btn.textContent = "🔊 Vocal Stem";
        btn.style.color = "#8b949e";
      }} else {{
        if (!card._audio) {{
          card._audio = new Audio(wavPath);
          video.onplay = () => {{ card._audio.currentTime = video.currentTime; card._audio.play(); }};
          video.onpause = () => card._audio.pause();
          video.onseeked = () => {{ card._audio.currentTime = video.currentTime; }};
        }}
        video.muted = true;
        btn.dataset.mode = "vocal";
        btn.textContent = "🎤 Vocal Stem";
        btn.style.color = "#7ee787";
      }}
    }}

    function toggleConfirm(actorId) {{
      const card = document.getElementById(`actor-card-${{actorId}}`);
      card.classList.toggle("confirmed");
      card.classList.remove("discarded");
      showToast(`Actor #${{actorId}} confirmed!`);
    }}

    function denyActor(actorId) {{
      const card = document.getElementById(`actor-card-${{actorId}}`);
      card.classList.toggle("discarded");
      card.classList.remove("confirmed");
      showToast(`Actor #${{actorId}} discarded.`);
    }}

    function showToast(msg) {{
      const el = document.getElementById("toast-msg");
      el.textContent = msg;
      el.style.display = "inline";
      setTimeout(() => {{ el.style.display = "none"; }}, 3500);
    }}

    function exportConfirmedActors() {{
      const result = {{
        exported_at: new Date().toISOString(),
        actors: {{}}
      }};

      Object.keys(actorsState).forEach(key => {{
        const actor = actorsState[key];
        const card = document.getElementById(`actor-card-${{actor.actor_id}}`);
        const isDiscarded = card.classList.contains("discarded");
        const customName = document.getElementById(`name-input-${{actor.actor_id}}`).value.trim();

        if (!isDiscarded) {{
          const activeClips = actor.samples.filter(s => !s.is_reserve);
          result.actors[key] = {{
            key: key,
            name: customName || actor.name || key,
            screen_time_formatted: actor.screen_time_formatted,
            confidence_pct: actor.confidence_pct,
            avatar_image: actor.avatar_image,
            active_clips: activeClips,
          }};
        }}
      }});

      const blob = new Blob([JSON.stringify(result, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "confirmed_actor_references.json";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      showToast("✅ Exported confirmed_actor_references.json!");
    }}
  </script>
</body>
</html>
"""
    html_path.write_text(html_code, encoding="utf-8")


def _render_clip_card(s: dict, actor_id: int) -> str:
    sim_str = f" | Sim: {s.get('similarity', 0.0):.2f}" if "similarity" in s else ""
    return f"""
    <div class="clip-card" id="card-{s['clip_id']}" draggable="true" ondragstart="handleDragStart(event, '{s['clip_id']}', {actor_id})" ondragend="handleDragEnd(event)">
      <div class="clip-header">
        <span class="clip-id">#{s['rank']} ({s['duration']:.1f}s)</span>
        <button class="delete-clip-btn" title="Delete clip and promote next reserve" onclick="deleteClip('{s['clip_id']}', {actor_id})">✕</button>
      </div>
      <video class="clip-video" controls preload="none" src="{s['mp4_path']}"></video>
      <div class="clip-quote">"{s['text']}"</div>
      <div class="clip-footer">
        <span class="motion-tag">Motion: {s['motion']}{sim_str}</span>
        <button class="toggle-audio-btn" onclick="toggleIsolatedVocal(this, '{s['wav_path']}')">🔊 Vocal Stem</button>
      </div>
    </div>
    """
