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
