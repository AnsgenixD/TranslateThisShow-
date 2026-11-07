#!/usr/bin/env python3
"""
benchmark_alibaba_vs_current.py - Benchmark Alibaba 3D-Speaker (CAM++) vs Our Engine.

Compares:
1. Pure Acoustic Speaker Verification & Discrimination across all 12 Cast Members.
2. Acoustic Cluster Attribution: How does Alibaba CAM++ assign whole-episode clusters
   compared to our Audio-Visual Hungarian & Multimodal Engine?
3. Where does acoustic-only AI fail (e.g. cross-talk, microphone filters, off-screen speakers)
   that necessitates visual reasoning?
"""

import os
import sys
import json
import glob
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str((PROJECT_ROOT / "CleanProject" / "3D-Speaker").resolve()))

from speakerlab.models.campplus.DTDNN import CAMPPlus


def load_campplus():
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    weights_path = Path.home() / ".cache/modelscope/models/damo--speech_campplus_sv_zh-cn_16k-common/snapshots/master/campplus_cn_common.bin"
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_embedding(model, audio_arr, sr=16000):
    wav = torch.from_numpy(audio_arr).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2 and wav.shape[0] > wav.shape[1]:
        wav = wav.t()
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    fbank = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    fbank = fbank.unsqueeze(0)
    with torch.no_grad():
        emb = model(fbank)
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    return emb.squeeze(0).numpy()


def main():
    print("=" * 80)
    print("🔬 ALIBABA 3D-SPEAKER (CAM++) VS OUR MULTI-MODAL ENGINE BENCHMARK")
    print("=" * 80)

    model = load_campplus()
    print("✅ Loaded Alibaba 3D-Speaker CAM++ (192-D DTDNN embedding model)")

    # 1. Extract reference embeddings for all 12 confirmed actors
    actors_dir = Path("output/visual_actors")
    actor_ids = [f"Person_{i:02d}" for i in range(1, 13)]
    actor_embs = {}

    for aid in actor_ids:
        wavs = sorted(glob.glob(str(actors_dir / aid / "sample_*.wav")))
        if not wavs:
            continue
        embs = []
        for w in wavs:
            try:
                y, sr = sf.read(w)
                if len(y) >= sr * 0.5:
                    embs.append(get_embedding(model, y, sr))
            except Exception:
                pass
        if embs:
            mean_emb = np.mean(embs, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            actor_embs[aid] = mean_emb

    print(f"👥 Extracted pure Alibaba CAM++ acoustic fingerprints for {len(actor_embs)} cast members.")

    # 2. Inter-Actor Discrimination Matrix
    print("\n📊 Inter-Actor Acoustic Discrimination Matrix (Alibaba CAM++ Cosine Similarity):")
    sim_matrix = []
    for a1 in actor_ids:
        row = [a1]
        for a2 in actor_ids:
            if a1 in actor_embs and a2 in actor_embs:
                s = float(np.dot(actor_embs[a1], actor_embs[a2]))
                row.append(f"{s:.2f}")
            else:
                row.append("-")
        sim_matrix.append(row)

    headers = ["Actor"] + [f"P_{i:02d}" for i in range(1, 13)]
    print(tabulate(sim_matrix, headers=headers, tablefmt="github"))

    # 3. Acoustic Cluster Attribution with Alibaba CAM++
    print("\n🎧 Benchmarking Whole-Episode Audio Clusters with Alibaba CAM++:")
    with open("output/diar_v2/pyannote_turns.json", "r", encoding="utf-8") as f:
        diar_data = json.load(f)
    turns = diar_data.get("exclusive_turns", diar_data.get("turns", []))
    clusters = sorted(diar_data["distinct_clusters"])

    with open("output/diar_v2/cluster_identity.json", "r", encoding="utf-8") as f:
        our_clusters = json.load(f)["clusters"]

    y_vocals, sr = sf.read("output/diar_v2/vocals_16k.wav")

    cluster_comparison = []

    for c in clusters:
        c_turns = [t for t in turns if t["cluster_id"] == c]
        # Concatenate audio up to 20s
        audio_chunks = []
        tot = 0.0
        for t in c_turns:
            s_i, e_i = int(t["start"] * sr), int(t["end"] * sr)
            if e_i > len(y_vocals): e_i = len(y_vocals)
            if s_i < e_i:
                audio_chunks.append(y_vocals[s_i:e_i])
                tot += (t["end"] - t["start"])
            if tot >= 20.0: break

        if not audio_chunks:
            continue

        c_audio = np.concatenate(audio_chunks)
        c_emb = get_embedding(model, c_audio, sr)

        # Match against cast members
        scores = {aid: float(np.dot(c_emb, emb)) for aid, emb in actor_embs.items()}
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        top_aid, top_score = sorted_scores[0]
        second_aid, second_score = sorted_scores[1]

        our_assignment = our_clusters.get(c, {}).get("assigned_actor", "UNKNOWN")
        our_type = our_clusters.get(c, {}).get("match_type", "")[:20]

        # Is Alibaba CAM++ confident?
        is_confident = (top_score >= 0.55 and (top_score - second_score) >= 0.08)
        alibaba_pred = top_aid if is_confident else f"{top_aid} (Weak: {top_score:.2f})"

        # Agreement check
        agrees = (our_assignment == top_aid)

        cluster_comparison.append([
            c,
            f"{tot:.1f}s",
            our_assignment,
            our_type,
            alibaba_pred,
            f"{top_score:.3f}",
            "✅ YES" if agrees else "⚠️ DIFF"
        ])

    comp_headers = [
        "Cluster", "Audio Time", "Our Engine Assignment", "Our Engine Logic",
        "Alibaba CAM++ Match", "CAM++ Score", "Agreement"
    ]
    print(tabulate(cluster_comparison, headers=comp_headers, tablefmt="github"))


if __name__ == "__main__":
    main()
