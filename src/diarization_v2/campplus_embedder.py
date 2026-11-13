#!/usr/bin/env python3
"""
campplus_embedder.py - Shared Alibaba 3D-Speaker (CAM++) Feature Extractor.

Loads the pretrained 192-D DTDNN model and computes L2-normalized speaker embeddings
for audio chunks, audio clusters, and actor voice references.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torchaudio
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPEAKER3D_DIR = REPO_ROOT / "3D-Speaker"
if str(SPEAKER3D_DIR) not in sys.path:
    sys.path.insert(0, str(SPEAKER3D_DIR))

from speakerlab.models.campplus.DTDNN import CAMPPlus

DEFAULT_WEIGHTS = Path.home() / ".cache/modelscope/models/damo--speech_campplus_sv_zh-cn_16k-common/snapshots/master/campplus_cn_common.bin"

_GLOBAL_CAMPPLUS_MODEL: Optional[CAMPPlus] = None


def load_campplus(weights_path: Optional[Path] = None) -> CAMPPlus:
    """Loads and caches the 192-D CAMPPlus model in eval mode on CPU."""
    global _GLOBAL_CAMPPLUS_MODEL
    if _GLOBAL_CAMPPLUS_MODEL is not None:
        return _GLOBAL_CAMPPLUS_MODEL

    path = weights_path or DEFAULT_WEIGHTS
    if not path.exists():
        raise FileNotFoundError(f"CAM++ weights not found at {path}. Please download via ModelScope.")

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    _GLOBAL_CAMPPLUS_MODEL = model
    return model


def get_campplus_embedding(model: CAMPPlus, audio_arr: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extracts a 192-D L2-normalized embedding from a raw mono audio numpy array."""
    if len(audio_arr) == 0:
        return np.zeros(192, dtype=np.float32)

    wav = torch.from_numpy(audio_arr).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2 and wav.shape[0] > wav.shape[1]:
        wav = wav.t()

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # Kaldi 80-bin FBank
    fbank = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=80)
    fbank = fbank - fbank.mean(dim=0, keepdim=True)
    fbank = fbank.unsqueeze(0)

    with torch.no_grad():
        emb = model(fbank)
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().numpy()


def extract_actor_reference_embeddings(
    model: CAMPPlus,
    actors_dir: str = "output/visual_actors",
    actor_ids: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """Computes the mean L2-normalized CAM++ embedding for each actor from purified sample wavs."""
    p_dir = Path(actors_dir)
    if actor_ids is None:
        actor_ids = [f"Person_{i:02d}" for i in range(1, 13)]

    actor_embs = {}
    for aid in actor_ids:
        wavs = sorted(glob.glob(str(p_dir / aid / "sample_*.wav")))
        if not wavs:
            continue
        embs = []
        for w in wavs:
            try:
                y, sr = sf.read(w)
                if len(y) >= sr * 0.4:
                    embs.append(get_campplus_embedding(model, y, sr))
            except Exception:
                pass
        if embs:
            mean_emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 1e-6:
                mean_emb = mean_emb / norm
            actor_embs[aid] = mean_emb

    return actor_embs


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    n1 = np.linalg.norm(emb1)
    n2 = np.linalg.norm(emb2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.dot(emb1, emb2) / (n1 * n2))
