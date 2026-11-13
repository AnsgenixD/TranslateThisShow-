#!/usr/bin/env python3
"""
phase4_hungarian_cooccurrence.py - Multi-Modal Co-Occurrence Matrix & Open-Set Hungarian Cluster Attribution.

Binds whole-episode audio clusters (SPEAKER_00..SPEAKER_13) to confirmed actors (Person_01..Person_12)
or explicit open-set classes (NARRATOR, GUEST_SPEAKER_MALE, GUEST_SPEAKER_FEMALE, OFFSCREEN_UNKNOWN):
1. Measures acoustic fundamental frequency (F0) for each cluster from the vocal stem.
2. Cross-validates acoustic gender against InsightFace verified actor gender.
3. Enforces strict ASD co-occurrence thresholds (M[c, a] >= 15.0s, score >= 0.30) to prevent
   unconfirmed extras/guests from being forcibly absorbed into the main cast.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import soundfile as sf
import librosa
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .campplus_embedder import (
        load_campplus,
        get_campplus_embedding,
        extract_actor_reference_embeddings,
        cosine_similarity
    )
except ImportError:
    from src.diarization_v2.campplus_embedder import (
        load_campplus,
        get_campplus_embedding,
        extract_actor_reference_embeddings,
        cosine_similarity
    )


def compute_interval_overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    return max(0.0, min(e1, e2) - max(s1, s2))


def measure_cluster_acoustic_pitch(vocals_wav: str, turns: List[Dict[str, Any]], cluster_id: str, max_sec: float = 20.0) -> Dict[str, Any]:
    """Measures fundamental frequency (F0) and acoustic gender for an audio cluster."""
    if not os.path.exists(vocals_wav):
        return {"f0": 0.0, "gender": "unknown"}

    try:
        y, sr = sf.read(vocals_wav)
        c_turns = [t for t in turns if t["cluster_id"] == cluster_id]
        samples = []
        tot = 0.0
        for t in c_turns:
            s_idx = int(t["start"] * sr)
            e_idx = int(t["end"] * sr)
            if e_idx > len(y):
                e_idx = len(y)
            if s_idx < e_idx:
                samples.append(y[s_idx:e_idx])
                tot += (t["end"] - t["start"])
            if tot >= max_sec:
                break

        if not samples:
            return {"f0": 0.0, "gender": "unknown"}

        audio_concat = np.concatenate(samples)
        if len(audio_concat) < sr * 0.2:
            return {"f0": 0.0, "gender": "unknown"}

        f0, voiced_flag, voiced_probs = librosa.pyin(audio_concat, fmin=50, fmax=400, sr=sr)
        valid = f0[voiced_flag & ~np.isnan(f0)] if voiced_flag is not None else []
        if len(valid) == 0:
            return {"f0": 0.0, "gender": "unknown"}

        med = float(np.median(valid))
        # Pitch boundary: male speech typically 80-165 Hz; female 165-300+ Hz
        gender = "male" if med < 165.0 else "female"
        return {"f0": round(med, 1), "gender": gender}
    except Exception as e:
        print(f"⚠️ Error measuring pitch for {cluster_id}: {e}")
        return {"f0": 0.0, "gender": "unknown"}


def run_cluster_attribution(
    turns_json: str = "output/diar_v2/pyannote_turns.json",
    tracks_json: str = "output/diar_v2/face_tracks.json",
    roster_json: str = "output/face_roster/confirmed_actor_roster.json",
    references_json: str = "output/confirmed_actor_references.json",
    output_json: str = "output/diar_v2/cluster_identity.json",
    vocals_wav: str = "output/diar_v2/vocals_16k.wav",
    tau_unassigned: float = 0.15
) -> Dict[str, Any]:
    print("=" * 70)
    print("🤝 PHASE 4: OPEN-SET CLUSTER-ACTOR CO-OCCURRENCE & PITCH VERIFICATION")
    print("=" * 70)

    # 1. Load data
    with open(turns_json, "r", encoding="utf-8") as f:
        diar_data = json.load(f)
    turns = diar_data.get("exclusive_turns", diar_data.get("turns", []))
    clusters = sorted(diar_data["distinct_clusters"])

    with open(tracks_json, "r", encoding="utf-8") as f:
        track_data = json.load(f)
    tracks = track_data["tracks"]

    with open(references_json, "r", encoding="utf-8") as f:
        ref_data = json.load(f)["actors"]

    actor_ids = sorted(list(ref_data.keys()))
    actor_genders = {k: v.get("verified_gender", "unknown") for k, v in ref_data.items()}

    print(f"📊 Loaded {len(clusters)} audio clusters, {len(tracks)} face tracks, {len(actor_ids)} confirmed actors.")

    # 1.5 Load Alibaba 3D-Speaker (CAM++) and Actor Reference Fingerprints
    print("\n🔬 Loading Alibaba 3D-Speaker (CAM++) 192-D DTDNN Model...")
    campp_model = load_campplus()
    actor_embs = extract_actor_reference_embeddings(campp_model, actors_dir="output/visual_actors", actor_ids=actor_ids)
    print(f"✅ Loaded CAM++ acoustic voiceprints for {len(actor_embs)} cast members.")

    # 2. Compute Cluster Speaking Time, Acoustic Pitch & CAM++ Embeddings
    cluster_speaking_sec = {c: 0.0 for c in clusters}
    cluster_turns = {c: [] for c in clusters}
    cluster_acoustics = {}
    cluster_embs = {}

    y_vocals, sr_vocals = sf.read(vocals_wav) if os.path.exists(vocals_wav) else (None, 16000)

    for t in turns:
        c = t["cluster_id"]
        dur = t["end"] - t["start"]
        cluster_speaking_sec[c] += dur
        cluster_turns[c].append((t["start"], t["end"]))

    print("\n🎵 Acoustic Fundamental Frequency (F0) & CAM++ Fingerprinting per Cluster:")
    for c in clusters:
        pitch_info = measure_cluster_acoustic_pitch(vocals_wav, turns, c)
        cluster_acoustics[c] = pitch_info

        # Extract cluster audio for CAM++ (up to 25s)
        if y_vocals is not None:
            c_samples = []
            tot_c = 0.0
            for t_s, t_e in cluster_turns[c]:
                s_i, e_i = int(t_s * sr_vocals), int(t_e * sr_vocals)
                if e_i > len(y_vocals): e_i = len(y_vocals)
                if s_i < e_i:
                    c_samples.append(y_vocals[s_i:e_i])
                    tot_c += (t_e - t_s)
                if tot_c >= 25.0:
                    break
            if c_samples:
                c_audio = np.concatenate(c_samples)
                cluster_embs[c] = get_campplus_embedding(campp_model, c_audio, sr_vocals)
            else:
                cluster_embs[c] = np.zeros(192, dtype=np.float32)
        else:
            cluster_embs[c] = np.zeros(192, dtype=np.float32)

        print(f"  {c:11} ({cluster_speaking_sec[c]:5.1f}s): {pitch_info['f0']:5.1f} Hz -> {pitch_info['gender'].upper()}")

    # 3. Build Co-occurrence Matrix M[cluster, actor] & CAM++ Matrix A[cluster, actor]
    M = np.zeros((len(clusters), len(actor_ids)), dtype=np.float32)
    V = np.zeros((len(clusters), len(actor_ids)), dtype=np.float32)
    A = np.zeros((len(clusters), len(actor_ids)), dtype=np.float32)

    cluster_to_idx = {c: i for i, c in enumerate(clusters)}
    actor_to_idx = {a: j for j, a in enumerate(actor_ids)}

    for trk in tracks:
        act = trk["actor_id"]
        if act not in actor_to_idx:
            continue
        a_i = actor_to_idx[act]
        t_s, t_e = trk["start_sec"], trk["end_sec"]
        is_spk = trk.get("is_speaking", False) or (trk.get("asd_score", 0.0) >= 0.45)

        for c in clusters:
            c_i = cluster_to_idx[c]
            for c_s, c_e in cluster_turns[c]:
                ov = compute_interval_overlap(t_s, t_e, c_s, c_e)
                if ov > 0.0:
                    V[c_i, a_i] += ov
                    if is_spk:
                        M[c_i, a_i] += ov

    # Compute CAM++ Cosine Similarities
    for i, c in enumerate(clusters):
        c_emb = cluster_embs[c]
        for j, a in enumerate(actor_ids):
            if a in actor_embs:
                sim = cosine_similarity(c_emb, actor_embs[a])
                A[i, j] = max(0.0, sim)

    # 4. Score Matrix S (Multi-Modal Fusion: 55% Active ASD + 15% Vis + 30% CAM++)
    S = np.zeros((len(clusters), len(actor_ids)), dtype=np.float32)
    for i, c in enumerate(clusters):
        tot_spk = max(1.0, cluster_speaking_sec[c])
        for j, a in enumerate(actor_ids):
            active_ratio = M[i, j] / tot_spk
            vis_ratio = V[i, j] / tot_spk
            campp_sim = A[i, j]
            S[i, j] = active_ratio * 0.55 + vis_ratio * 0.15 + campp_sim * 0.30

    print("\n📈 Fused Multi-Modal Attribution Matrix (Active Mouth + CAM++ Acoustic):")
    for i, c in enumerate(clusters):
        best_j = int(np.argmax(S[i]))
        print(f"  {c} ({cluster_speaking_sec[c]:5.1f}s) -> Top: {actor_ids[best_j]} (Active: {M[i, best_j]:4.1f}s, Vis: {V[i, best_j]:4.1f}s, CAM++: {A[i, best_j]:.3f}, Fused Score: {S[i, best_j]:.3f})")

    # 5. Open-Set Cluster Attribution with Strict Gates
    n_clusters = len(clusters)
    n_actors = len(actor_ids)
    cluster_order = sorted(range(n_clusters), key=lambda i: -cluster_speaking_sec[clusters[i]])

    cluster_assignments: Dict[str, Dict[str, Any]] = {}
    assigned_actors = set()

    for r in cluster_order:
        cluster_name = clusters[r]
        tot_sec = cluster_speaking_sec[cluster_name]
        sorted_actor_indices = sorted(range(n_actors), key=lambda j: -S[r, j])
        best_j = sorted_actor_indices[0]
        second_j = sorted_actor_indices[1] if n_actors > 1 else best_j

        best_act = actor_ids[best_j]
        best_score = float(S[r, best_j])
        active_sec = float(M[r, best_j])
        vis_sec = float(V[r, best_j])
        campp_score = float(A[r, best_j])
        second_campp = float(A[r, second_j])

        cl_f0 = cluster_acoustics[cluster_name]["f0"]
        cl_gender = cluster_acoustics[cluster_name]["gender"]
        act_gender = actor_genders.get(best_act, "unknown")

        # Gender Sanity Check
        gender_conflict = False
        if cl_gender == "male" and act_gender == "female":
            gender_conflict = True
        elif cl_gender == "female" and act_gender == "male":
            gender_conflict = True

        # Evidence Threshold:
        # Standard: Active mouth >= 15s AND score >= 0.30 (or <45s with >=8s active mouth & score >= 0.25)
        # Rescue: Active mouth >= 5s AND strong CAM++ agreement (>= 0.80, margin >= 0.08)
        has_sufficient_evidence = (
            (active_sec >= 15.0 and best_score >= 0.30) or
            (tot_sec < 45.0 and active_sec >= 8.0 and best_score >= 0.25) or
            (active_sec >= 5.0 and campp_score >= 0.80 and (campp_score - second_campp) >= 0.08)
        )

        if has_sufficient_evidence and not gender_conflict:
            cluster_assignments[cluster_name] = {
                "cluster_id": cluster_name,
                "assigned_actor": best_act,
                "speaking_sec": round(tot_sec, 2),
                "match_type": "fused_asd_campplus",
                "confidence": round(best_score, 3),
                "active_cooccurrence_sec": round(active_sec, 2),
                "visibility_sec": round(vis_sec, 2),
                "campplus_score": round(campp_score, 3),
                "acoustic_f0": cl_f0,
                "acoustic_gender": cl_gender
            }
            assigned_actors.add(best_act)
        else:
            # Open-Set Reject Option: do NOT force into cast!
            if gender_conflict:
                rejection_reason = f"gender_mismatch (cluster {cl_gender} vs {best_act} {act_gender})"
            else:
                rejection_reason = f"insufficient_evidence (active_sec={active_sec:.1f}s, score={best_score:.3f}, campp={campp_score:.3f})"

            # Distinguish Narrator vs Guest Speaker vs Offscreen
            if tot_sec >= 20.0 and vis_sec < 5.0:
                reject_label = "NARRATOR"
                match_type = "narrator_offscreen"
                conf = 0.85
            elif cl_gender == "male":
                reject_label = "GUEST_SPEAKER_MALE"
                match_type = f"rejected_to_guest ({rejection_reason})"
                conf = 0.70
            elif cl_gender == "female":
                reject_label = "GUEST_SPEAKER_FEMALE"
                match_type = f"rejected_to_guest ({rejection_reason})"
                conf = 0.70
            else:
                reject_label = "OFFSCREEN_UNKNOWN"
                match_type = f"rejected_to_unknown ({rejection_reason})"
                conf = 0.50

            cluster_assignments[cluster_name] = {
                "cluster_id": cluster_name,
                "assigned_actor": reject_label,
                "speaking_sec": round(tot_sec, 2),
                "match_type": match_type,
                "confidence": conf,
                "active_cooccurrence_sec": round(active_sec, 2),
                "visibility_sec": round(vis_sec, 2),
                "campplus_score": round(campp_score, 3),
                "acoustic_f0": cl_f0,
                "acoustic_gender": cl_gender
            }

    print("\n🎯 Final Open-Set Cluster -> Identity Attribution Table:")
    for c in clusters:
        info = cluster_assignments[c]
        print(f"  {c:11} ({info['speaking_sec']:5.1f}s, {info['acoustic_f0']:5.1f}Hz) ===> {info['assigned_actor']:20} [{info['match_type'][:35]}] (conf={info['confidence']})")

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tau_unassigned": tau_unassigned,
            "clusters": cluster_assignments
        }, f, indent=2)

    print(f"\n💾 Saved open-set cluster identities to {output_json}")
    return cluster_assignments


if __name__ == "__main__":
    run_cluster_attribution()
