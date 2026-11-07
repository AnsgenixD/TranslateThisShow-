#!/usr/bin/env python3
"""
phase1_pyannote.py - Whole-Episode Speaker Diarization using pyannote community-1.

1. Resamples vocals to 16kHz mono -> output/diar_v2/vocals_16k.wav.
2. Runs pyannote/speaker-diarization-community-1 with min_speakers=6, max_speakers=20.
3. Exports both standard RTTM and exclusive (non-overlapping) RTTM.
4. Caches structured turn manifest to output/diar_v2/pyannote_turns.json.
"""

import os
import sys
import json
import time
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
import librosa
from pyannote.audio import Pipeline


def ensure_16k_mono(input_wav: str, output_wav: str) -> str:
    out_p = Path(output_wav)
    if out_p.exists() and out_p.stat().st_size > 10000:
        print(f"✅ Found existing 16kHz audio: {output_wav}")
        return str(out_p)

    print(f"🎵 Resampling {input_wav} to 16kHz mono...")
    audio, sr = librosa.load(input_wav, sr=16000, mono=True)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_p), audio, 16000)
    print(f"💾 Saved {len(audio)/16000:.1f}s to {output_wav}")
    return str(out_p)


def run_pyannote_diarization(
    vocals_wav: str,
    output_dir: str = "output/diar_v2",
    min_speakers: int = 6,
    max_speakers: int = 20,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rttm_path = out_dir / "pyannote.rttm"
    turns_json_path = out_dir / "pyannote_turns.json"

    # Step 1: Resample
    wav_16k = ensure_16k_mono(vocals_wav, str(out_dir / "vocals_16k.wav"))

    # Step 2: Load Pipeline
    token = os.environ.get("HF_TOKEN")
    print(f"🚀 Loading pyannote/speaker-diarization-community-1 on {device}...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token
    )
    
    # Try sending to device
    try:
        if device == "mps":
            pipeline.to(torch.device("mps"))
        elif device == "cuda":
            pipeline.to(torch.device("cuda"))
    except Exception as ex:
        print(f"⚠️ Could not move pipeline to {device} ({ex}), using CPU.")

    print(f"🎙️ Running full-episode diarization (min_speakers={min_speakers}, max_speakers={max_speakers})...")
    t0 = time.time()
    diarization = pipeline(
        wav_16k,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    elapsed = time.time() - t0
    print(f"⏱️ Diarization completed in {elapsed:.1f}s")

    # Step 3: Write standard and exclusive RTTMs + embeddings
    if hasattr(diarization, "speaker_diarization"):
        std_ann = diarization.speaker_diarization
        excl_ann = diarization.exclusive_speaker_diarization
        spk_embs = diarization.speaker_embeddings
    else:
        std_ann = diarization
        excl_ann = None
        spk_embs = None

    with open(rttm_path, "w", encoding="utf-8") as f:
        std_ann.write_rttm(f)
    print(f"💾 Saved standard RTTM to {rttm_path}")

    excl_rttm_path = out_dir / "pyannote_exclusive.rttm"
    if excl_ann is not None:
        with open(excl_rttm_path, "w", encoding="utf-8") as f:
            excl_ann.write_rttm(f)
        print(f"💾 Saved exclusive RTTM to {excl_rttm_path}")

    if spk_embs is not None:
        np.save(out_dir / "pyannote_speaker_embeddings.npy", spk_embs)
        print(f"💾 Saved speaker embeddings {spk_embs.shape} to {out_dir / 'pyannote_speaker_embeddings.npy'}")

    # Step 4: Extract structured turns
    turns = []
    clusters = set()
    for turn, _, speaker in std_ann.itertracks(yield_label=True):
        clusters.add(speaker)
        turns.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3),
            "cluster_id": speaker
        })

    exclusive_turns = []
    if excl_ann is not None:
        for turn, _, speaker in excl_ann.itertracks(yield_label=True):
            exclusive_turns.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "duration": round(turn.end - turn.start, 3),
                "cluster_id": speaker
            })

    print(f"📊 Discovered {len(clusters)} distinct speaker clusters across {len(turns)} speech turns.")

    with open(turns_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_turns": len(turns),
            "total_exclusive_turns": len(exclusive_turns),
            "distinct_clusters": sorted(list(clusters)),
            "turns": turns,
            "exclusive_turns": exclusive_turns
        }, f, indent=2)

    print(f"💾 Saved turn manifest to {turns_json_path}")
    return turns_json_path


if __name__ == "__main__":
    vocals = "output/ep1_lower_30m_stems/htdemucs/ep1_lower_30m_24k/vocals.wav"
    if len(sys.argv) > 1:
        vocals = sys.argv[1]
    run_pyannote_diarization(vocals)
