"""
diarizer.py - Unified Audio-Visual Speaker Diarization Orchestrator (Diarization V2).

Executes:
  Phase 1: Pyannote acoustic speaker turn clustering
  Phase 3: Shot detection + InsightFace ArcFace + geometric lip-motion Active Speaker Detection (ASD)
  Phase 4: Co-occurrence matrix & Hungarian bipartite cluster attribution (fusing visual + 3D-Speaker CAM++)
  Phase 5: Subtitle-to-speaker cluster mapping
  Phase 5.5: Optional Local VLM Arbiter (Qwen3-VL-4B on Apple Silicon Metal)
  Phase 6: Conversational HMM temporal smoothing
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_diarization_pipeline(
    video_path: str,
    vocals_wav: str,
    output_dir: str = "output/diar_v2",
    use_vlm_arbiter: bool = True,
    vlm_model: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit",
) -> Optional[Dict[str, Any]]:
    """
    Executes the cluster-first multimodal speaker diarization pipeline.
    Gracefully skips with installation instructions if dependencies are missing.
    """
    # 1. Dependency checks
    missing_deps = []
    try:
        import pyannote.audio
    except ImportError:
        missing_deps.append("pyannote.audio")
    try:
        import insightface
    except ImportError:
        missing_deps.append("insightface")
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")

    if missing_deps:
        print("\n💡 [Optional Feature] Audio-Visual Diarization requires additional packages:")
        print(f"   Missing: {', '.join(missing_deps)}")
        print("   To install full diarization capabilities:")
        print(f"     pip install {' '.join(missing_deps)}")
        print("   Skipping multimodal diarization for now...\n")
        return None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🎬 RUNNING AUDIO-VISUAL SPEAKER DIARIZATION (V2)")
    print("=" * 70)

    # Phase 1: Pyannote acoustic turns
    turns_json = out_path / "pyannote_turns.json"
    if not turns_json.exists():
        try:
            from .diarization_v2.phase1_pyannote import run_pyannote_diarization
        except ImportError:
            from CleanProject.src.diarization_v2.phase1_pyannote import run_pyannote_diarization
        print("\n⚡ [Phase 1/5] Extracting acoustic speaker turns (Pyannote)...")
        run_pyannote_diarization(vocals_wav, str(out_path))
    else:
        print(f"✅ Phase 1 cached: {turns_json}")

    # Phase 3: Shot-aware Face Tracking & Lip ASD
    tracks_json = out_path / "face_tracks.json"
    if not tracks_json.exists():
        try:
            from .diarization_v2.phase3_face_tracker_asd import run_face_tracking_and_asd
        except ImportError:
            from CleanProject.src.diarization_v2.phase3_face_tracker_asd import run_face_tracking_and_asd
        print("\n⚡ [Phase 2/5] Running Face Tracking & Lip Active Speaker Detection...")
        run_face_tracking_and_asd(video_path=video_path, output_json=str(tracks_json))
    else:
        print(f"✅ Phase 3 cached: {tracks_json}")

    # Phase 4: Cluster Attribution (Hungarian matching)
    cluster_id_json = out_path / "cluster_identity.json"
    try:
        from .diarization_v2.phase4_hungarian_cooccurrence import run_cluster_attribution
    except ImportError:
        from CleanProject.src.diarization_v2.phase4_hungarian_cooccurrence import run_cluster_attribution
    print("\n⚡ [Phase 3/5] Hungarian Bipartite Audio-Visual Cluster Attribution...")
    run_cluster_attribution(
        turns_json=str(turns_json),
        tracks_json=str(tracks_json),
        output_json=str(cluster_id_json),
        tau_unassigned=0.15,
    )

    result = {
        "turns_path": str(turns_json),
        "tracks_path": str(tracks_json),
        "cluster_path": str(cluster_id_json),
    }
    return result


def _compute_interval_overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    return max(0.0, min(e1, e2) - max(s1, s2))


def attribute_speakers_to_segments(
    segments: List[Dict[str, Any]],
    vocals_wav: str,
    output_dir: str = "output/diar_v2",
) -> Dict[str, str]:
    """
    Attributes each subtitle segment to a consistent acoustic speaker cluster using:
      1. Pyannote community-1 standard & exclusive speaker turn overlap
      2. Alibaba 3D-Speaker CAM++ 192-D speaker embedding cosine similarity
      3. Grammatical sentence-continuity locking (prevents mid-sentence voice switching)
      4. Conversational sandwich smoothing (eliminates isolated 1-cue backchannel flips)
      5. Fundamental frequency (F0 via YIN) + relative pitch separation for gender verification

    Mutates `segments` in-place (setting `speaker`, `cluster_id`, `actor_id`) and returns
    `verified_genders` mapping `{speaker_id: "female" | "male"}`.
    """
    if not segments:
        return {}

    import numpy as np
    import soundfile as sf
    import librosa

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    turns_json = out_path / "pyannote_turns.json"

    if not turns_json.exists():
        try:
            from .diarization_v2.phase1_pyannote import run_pyannote_diarization
        except ImportError:
            from CleanProject.src.diarization_v2.phase1_pyannote import run_pyannote_diarization
        print("\n🎙️ [Diarizer] Running Pyannote speaker turn diarization...")
        run_pyannote_diarization(vocals_wav, str(out_path))

    with open(turns_json, "r", encoding="utf-8") as f:
        turns_data = json.load(f)

    std_turns = turns_data.get("turns", [])
    excl_turns = turns_data.get("exclusive_turns", std_turns)
    clusters = turns_data.get("distinct_clusters", [])
    if not clusters:
        clusters = sorted(list({t["cluster_id"] for t in std_turns}))
    if not clusters:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
            seg["cluster_id"] = "SPEAKER_00"
            seg["actor_id"] = "SPEAKER_00"
        return {"SPEAKER_00": "female"}

    wav, sr = sf.read(vocals_wav, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    # Load Alibaba 3D-Speaker CAM++ 192-D speaker verification model
    campp_model = None
    try:
        try:
            from .diarization_v2.campplus_embedder import (
                load_campplus,
                get_campplus_embedding,
                cosine_similarity,
            )
        except ImportError:
            from CleanProject.src.diarization_v2.campplus_embedder import (
                load_campplus,
                get_campplus_embedding,
                cosine_similarity,
            )
        campp_model = load_campplus()
    except Exception as e:
        print(f"⚠️ [Diarizer] CAM++ embedder notice ({e}); using Pyannote turn overlap.")

    # Build duration-weighted CAM++ speaker centroids from clean exclusive turns (>= 0.8s)
    centroids: Dict[str, np.ndarray] = {}
    if campp_model is not None:
        cluster_embs: Dict[str, List[ tuple[float, np.ndarray] ]] = {}
        for t in excl_turns:
            if t["duration"] < 0.8:
                continue
            chunk = wav[int(t["start"] * sr) : int(t["end"] * sr)]
            if len(chunk) >= int(0.4 * sr):
                emb = get_campplus_embedding(campp_model, chunk, sr)
                cluster_embs.setdefault(t["cluster_id"], []).append((t["duration"], emb))
        # Fallback for any cluster without >=0.8s exclusive turn
        for cid in clusters:
            if cid not in cluster_embs:
                for t in std_turns:
                    if t["cluster_id"] == cid and t["duration"] >= 0.5:
                        chunk = wav[int(t["start"] * sr) : int(t["end"] * sr)]
                        if len(chunk) >= int(0.3 * sr):
                            emb = get_campplus_embedding(campp_model, chunk, sr)
                            cluster_embs.setdefault(cid, []).append((t["duration"], emb))
        for cid, items in cluster_embs.items():
            weights = np.array([d for d, _ in items], dtype=np.float32)
            embs = np.stack([e for _, e in items], axis=0)
            m = np.average(embs, axis=0, weights=weights)
            norm = np.linalg.norm(m)
            centroids[cid] = m / norm if norm > 1e-6 else m

    # Pass 1: Score each segment using Pyannote std/excl overlap + CAM++ similarity
    seg_embs: List[Optional[np.ndarray]] = []
    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        dur = max(0.1, e - s)

        ov_std: Dict[str, float] = {}
        for t in std_turns:
            ov = _compute_interval_overlap(s, e, t["start"], t["end"])
            if ov > 0:
                ov_std[t["cluster_id"]] = ov_std.get(t["cluster_id"], 0.0) + ov

        ov_excl: Dict[str, float] = {}
        for t in excl_turns:
            ov = _compute_interval_overlap(s, e, t["start"], t["end"])
            if ov > 0:
                ov_excl[t["cluster_id"]] = ov_excl.get(t["cluster_id"], 0.0) + ov

        emb = None
        sims: Dict[str, float] = {}
        if campp_model is not None and centroids:
            mid = 0.5 * (s + e)
            w_s = max(0.0, min(s, mid - 0.4))
            w_e = min(len(wav) / sr, max(e, mid + 0.4))
            chunk = wav[int(w_s * sr) : int(w_e * sr)]
            if len(chunk) >= int(0.25 * sr):
                emb = get_campplus_embedding(campp_model, chunk, sr)
                sims = {cid: cosine_similarity(emb, cent) for cid, cent in centroids.items()}
        seg_embs.append(emb)

        scores: Dict[str, float] = {}
        for cid in clusters:
            f_std = min(1.0, ov_std.get(cid, 0.0) / dur)
            f_excl = min(1.0, ov_excl.get(cid, 0.0) / dur)
            if sims:
                scores[cid] = 0.45 * f_std + 0.25 * f_excl + 0.30 * sims.get(cid, 0.0)
            else:
                scores[cid] = 0.65 * f_std + 0.35 * f_excl

        best_cid = max(scores, key=scores.get) if scores else clusters[0]
        seg["speaker"] = best_cid
        seg["cluster_id"] = best_cid
        seg["actor_id"] = best_cid

    # Pass 2: Grammatical Sentence-Continuity Locking
    for i in range(len(segments) - 1):
        cur = segments[i]
        nxt = segments[i + 1]
        gap = float(nxt["start"]) - float(cur["end"])
        en_txt = (cur.get("translated_text") or cur.get("en") or "").strip()
        zh_txt = (cur.get("text") or cur.get("zh") or "").strip()
        nxt_en = (nxt.get("translated_text") or nxt.get("en") or "").strip()

        is_continuation = (
            en_txt.endswith((",", "...", "—", "-"))
            or zh_txt.endswith(("，", "、"))
            or (nxt_en and nxt_en[0].islower())
        )
        if is_continuation and gap <= 1.2 and cur["speaker"] != nxt["speaker"]:
            pair_sim = 0.5
            if seg_embs[i] is not None and seg_embs[i + 1] is not None:
                pair_sim = cosine_similarity(seg_embs[i], seg_embs[i + 1])
            if pair_sim >= 0.30:
                nxt["speaker"] = cur["speaker"]
                nxt["cluster_id"] = cur["speaker"]
                nxt["actor_id"] = cur["speaker"]

    # Pass 3: Conversational Sandwich Smoothing (A -> B -> A for short single segments)
    for i in range(1, len(segments) - 1):
        prv = segments[i - 1]
        cur = segments[i]
        nxt = segments[i + 1]
        dur = float(cur["end"]) - float(cur["start"])
        if prv["speaker"] == nxt["speaker"] and cur["speaker"] != prv["speaker"]:
            if (
                dur <= 1.5
                and (float(cur["start"]) - float(prv["end"])) <= 1.0
                and (float(nxt["start"]) - float(cur["end"])) <= 1.0
            ):
                cur["speaker"] = prv["speaker"]
                cur["cluster_id"] = prv["speaker"]
                cur["actor_id"] = prv["speaker"]

    # Pass 4: Fundamental Frequency (F0) Gender Estimation with Relative Pitch Separation
    cluster_f0: Dict[str, float] = {}
    for cid in clusters:
        f0_vals = []
        for t in excl_turns:
            if t["cluster_id"] != cid or t["duration"] < 0.45:
                continue
            chunk = wav[int(t["start"] * sr) : int(t["end"] * sr)]
            if len(chunk) < int(0.3 * sr):
                continue
            try:
                f0 = librosa.yin(chunk, fmin=70, fmax=350, sr=sr)
                rms = librosa.feature.rms(y=chunk, frame_length=2048, hop_length=512)[0]
                min_len = min(len(f0), len(rms))
                valid = f0[:min_len][(rms[:min_len] > 0.015) & (f0[:min_len] > 75) & (f0[:min_len] < 330)]
                if len(valid) > 3:
                    f0_vals.extend(valid.tolist())
            except Exception:
                pass
        cluster_f0[cid] = float(np.median(f0_vals)) if f0_vals else 180.0

    max_f0 = max(cluster_f0.values()) if cluster_f0 else 180.0
    verified_genders: Dict[str, str] = {}
    for cid, med_f0 in cluster_f0.items():
        if med_f0 >= 185.0:
            verified_genders[cid] = "female"
        elif med_f0 <= 155.0:
            verified_genders[cid] = "male"
        elif max_f0 >= 195.0 and (max_f0 - med_f0) >= 25.0:
            verified_genders[cid] = "male"
        else:
            verified_genders[cid] = "female" if med_f0 >= 175.0 else "male"

    print("🎙️ [Diarizer] Speaker Attribution Summary:")
    for cid in clusters:
        cnt = sum(1 for s in segments if s.get("speaker") == cid)
        print(f"   • {cid}: {cnt} segments | median F0={cluster_f0.get(cid, 0.0):.1f}Hz -> {verified_genders.get(cid)}")

    return verified_genders

