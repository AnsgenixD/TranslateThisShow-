#!/usr/bin/env python3
"""
phase5_subtitle_mapper.py - Binds Official Subtitle Lines to Audio Clusters and Candidate Actors.

Maps each of the 791 official subtitle lines to its primary audio diarization cluster
and computes multi-modal emission log-probabilities across candidate actor labels.
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "CleanProject") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "CleanProject"))

try:
    from .campplus_embedder import (
        load_campplus,
        get_campplus_embedding,
        extract_actor_reference_embeddings,
        cosine_similarity
    )
except ImportError:
    from CleanProject.src.diarization_v2.campplus_embedder import (
        load_campplus,
        get_campplus_embedding,
        extract_actor_reference_embeddings,
        cosine_similarity
    )


def parse_srt(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\d+\n|\Z)"
    )
    entries = []
    for m in pattern.finditer(text):
        idx, s_str, e_str, content = m.groups()
        h, mn, s_ms = s_str.split(":")
        s, ms = s_ms.split(",")
        start_sec = int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        h, mn, s_ms = e_str.split(":")
        s, ms = s_ms.split(",")
        end_sec = int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        entries.append({
            "orig_id": int(idx),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(end_sec - start_sec, 3),
            "text": content.strip().replace("\n", " ")
        })
    return entries


def compute_interval_overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    return max(0.0, min(e1, e2) - max(s1, s2))


def map_subtitles_to_clusters(
    cn_srt: str,
    en_srt: str,
    turns_json: str = "output/diar_v2/pyannote_turns.json",
    cluster_id_json: str = "output/diar_v2/cluster_identity.json",
    tracks_json: str = "output/diar_v2/face_tracks.json",
    vocals_wav: str = "output/diar_v2/vocals_16k.wav",
    output_json: str = "output/diar_v2/mapped_subtitles_raw.json",
    max_time: float = 1800.0
) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("🗺️ PHASE 5: SUBTITLE TO CLUSTER MAPPING & EMISSION SCORING")
    print("=" * 70)

    # 1. Parse official subtitles
    cn_entries = [e for e in parse_srt(cn_srt) if e["start"] <= max_time]
    en_entries = [e for e in parse_srt(en_srt) if e["start"] <= max_time]

    en_dict = {e["orig_id"]: e for e in en_entries}
    print(f"📖 Loaded {len(cn_entries)} official subtitle lines within {max_time}s.")

    # 2. Load diarization turns & cluster identities
    with open(turns_json, "r", encoding="utf-8") as f:
        diar_data = json.load(f)
    turns = diar_data.get("exclusive_turns", diar_data.get("turns", []))

    with open(cluster_id_json, "r", encoding="utf-8") as f:
        cluster_map = json.load(f)["clusters"]

    # 3. Load face tracks
    with open(tracks_json, "r", encoding="utf-8") as f:
        track_data = json.load(f)
    tracks = track_data["tracks"]

    # 3.5 Load CAM++ Model and Actor References for Line-Level Acoustic Scoring
    vocals_path = Path(vocals_wav)
    y_vocals, sr_vocals = sf.read(str(vocals_path)) if vocals_path.exists() else (None, 16000)
    print("🔬 Loading Alibaba 3D-Speaker (CAM++) for line-level acoustic verification...")
    campp_model = load_campplus()
    actor_embs = extract_actor_reference_embeddings(campp_model, actors_dir="output/visual_actors")
    print(f"✅ Loaded CAM++ references for {len(actor_embs)} cast members.")

    mapped_lines = []

    for idx, c_line in enumerate(cn_entries, 1):
        oid = c_line["orig_id"]
        e_line = en_dict.get(oid, {"text": ""})
        s = c_line["start"]
        e = c_line["end"]
        dur = max(0.2, e - s)

        # A. Find cluster overlaps
        cluster_overlaps: Dict[str, float] = {}
        for t in turns:
            ov = compute_interval_overlap(s, e, t["start"], t["end"])
            if ov > 0.0:
                cid = t["cluster_id"]
                cluster_overlaps[cid] = cluster_overlaps.get(cid, 0.0) + ov

        if cluster_overlaps:
            sorted_ov = sorted(cluster_overlaps.items(), key=lambda x: -x[1])
            best_cluster, best_ov_sec = sorted_ov[0]
            ov_fraction = round(best_ov_sec / dur, 3)
            possible_overlap = (len(sorted_ov) > 1 and sorted_ov[1][1] / dur >= 0.30)
        else:
            best_cluster = "UNKNOWN_CLUSTER"
            best_ov_sec = 0.0
            ov_fraction = 0.0
            possible_overlap = False

        # Cluster-assigned actor
        cluster_info = cluster_map.get(best_cluster, {})
        cluster_actor = cluster_info.get("assigned_actor", "OFFSCREEN_UNKNOWN")
        cluster_conf = cluster_info.get("confidence", 0.0)

        # B. Direct Line-Level ASD Evidence from Face Tracks
        direct_asd_actor = None
        max_asd_score = 0.0
        for trk in tracks:
            ov = compute_interval_overlap(s, e, trk["start_sec"], trk["end_sec"])
            if ov >= 0.20:
                if trk.get("is_speaking") and trk["actor_id"] != "UNKNOWN_FACE":
                    if trk["asd_score"] > max_asd_score:
                        max_asd_score = trk["asd_score"]
                        direct_asd_actor = trk["actor_id"]

        # C. Line-Level CAM++ Acoustic Fingerprint
        campp_top_actor = None
        campp_top_score = 0.0
        if y_vocals is not None and dur >= 0.6:
            s_i = int(s * sr_vocals)
            e_i = int(e * sr_vocals)
            if e_i > len(y_vocals): e_i = len(y_vocals)
            if (e_i - s_i) >= int(sr_vocals * 0.3):
                line_audio = y_vocals[s_i:e_i]
                line_emb = get_campplus_embedding(campp_model, line_audio, sr_vocals)
                line_scores = {aid: cosine_similarity(line_emb, a_emb) for aid, a_emb in actor_embs.items()}
                if line_scores:
                    sorted_scores = sorted(line_scores.items(), key=lambda x: -x[1])
                    campp_top_actor, campp_top_score = sorted_scores[0]
                    campp_top_score = round(float(campp_top_score), 3)

        # Decision rule:
        # If strong on-screen active speaker exists (max_asd_score >= 0.65), trust face!
        # Otherwise, trust the cluster identity!
        if direct_asd_actor is not None and max_asd_score >= 0.65:
            assigned_actor = direct_asd_actor
            match_type = "asd_direct"
            confidence = round(max_asd_score, 2)
        elif best_cluster != "UNKNOWN_CLUSTER" and cluster_actor == "NARRATOR":
            assigned_actor = "NARRATOR"
            match_type = "narrator_cluster"
            confidence = 0.85
        elif best_cluster != "UNKNOWN_CLUSTER" and ov_fraction >= 0.20:
            assigned_actor = cluster_actor
            match_type = "asd_cluster" if cluster_actor.startswith("Person_") else "open_set_cluster"
            confidence = round(min(0.95, cluster_conf * 0.7 + ov_fraction * 0.3), 2)
        else:
            # Low overlap or unconfirmed speech turn: open-set reject option
            assigned_actor = cluster_actor if cluster_actor != "UNKNOWN_CLUSTER" else "OFFSCREEN_UNKNOWN"
            match_type = "open_set_unassigned"
            confidence = 0.40

        mapped_lines.append({
            "index": idx,
            "orig_id": oid,
            "start": s,
            "end": e,
            "duration": dur,
            "zh": c_line["text"],
            "en": e_line["text"],
            "cluster_id": best_cluster,
            "cluster_actor": cluster_actor,
            "cluster_overlap_sec": round(best_ov_sec, 2),
            "cluster_overlap_frac": ov_fraction,
            "possible_overlap": possible_overlap,
            "direct_asd_actor": direct_asd_actor,
            "direct_asd_score": round(max_asd_score, 2),
            "campp_top_actor": campp_top_actor,
            "campp_top_score": campp_top_score,
            "actor_id": assigned_actor,
            "actor_name": assigned_actor,
            "match_type": match_type,
            "confidence": confidence
        })

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(mapped_lines, f, indent=2)

    print(f"✅ Mapped {len(mapped_lines)} lines to clusters and actors. Saved to {output_json}")
    return mapped_lines


if __name__ == "__main__":
    cn = "output/sample_source.srt"
    en = "output/sample_translated.srt"
    map_subtitles_to_clusters(cn, en)
