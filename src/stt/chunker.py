"""
chunker.py - Diarization-Aware & Overlap-Preserving Audio Chunker.

Splits dialogue into speaker-homogeneous audio chunks using Pyannote turns.
Handles overlapping speech by maintaining parallel chunks for concurrent speakers.
Applies acoustic padding and sensible duration limits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DiarizationChunker:
    """
    Transforms Pyannote acoustic turns into speaker-coherent ASR chunks.
    """

    def __init__(
        self,
        turns_path: Optional[str] = "output/diar_v2/pyannote_turns.json",
        max_chunk_duration: float = 12.0,
        min_chunk_duration: float = 0.5,
        merge_gap_threshold: float = 0.65,
        pad_onset: float = 0.15,
        pad_offset: float = 0.15,
    ):
        self.turns_path = Path(turns_path) if turns_path else None
        self.max_chunk_duration = max_chunk_duration
        self.min_chunk_duration = min_chunk_duration
        self.merge_gap_threshold = merge_gap_threshold
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.turns: List[Dict[str, Any]] = []

        if self.turns_path and self.turns_path.exists():
            self.load_turns()

    def load_turns(self) -> None:
        """Loads turns from pyannote_turns.json."""
        try:
            data = json.loads(self.turns_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.turns = data.get("turns", [])
            elif isinstance(data, list):
                self.turns = data
            self.turns.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
        except Exception as e:
            print(f"⚠️ Error loading turns from {self.turns_path}: {e}")
            self.turns = []

    def get_chunks(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        total_audio_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Builds speaker-homogeneous chunks within the time window [start_time, end_time].
        Merges consecutive turns of the same speaker if separated by <= merge_gap_threshold.
        Maintains independent parallel chunks for overlapping speakers.
        """
        if not self.turns:
            # Fallback if no diarization turns exist: chunk by fixed windows
            dur = end_time - start_time if end_time else 30.0
            return [{
                "chunk_id": 0,
                "start": start_time,
                "end": end_time or (start_time + dur),
                "pad_start": max(0.0, start_time - self.pad_onset),
                "pad_end": (end_time or (start_time + dur)) + self.pad_offset,
                "cluster_id": "SPEAKER_UNKNOWN",
            }]

        # Filter turns falling into the requested window
        filtered: List[Dict[str, Any]] = []
        for t in self.turns:
            t_s = float(t.get("start", 0.0))
            t_e = float(t.get("end", 0.0))
            if end_time is not None and t_s >= end_time:
                continue
            if t_e <= start_time:
                continue

            # Clip to window boundaries
            c_s = max(start_time, t_s)
            c_e = min(end_time, t_e) if end_time is not None else t_e
            if c_e > c_s + 0.05:
                cluster = t.get("cluster_id") or t.get("speaker") or "SPEAKER_00"
                filtered.append({
                    "start": round(c_s, 3),
                    "end": round(c_e, 3),
                    "cluster_id": cluster,
                })

        # Merge adjacent turns of the same speaker
        merged: List[Dict[str, Any]] = []
        for t in filtered:
            if not merged:
                merged.append(dict(t))
                continue

            prev = merged[-1]
            gap = t["start"] - prev["end"]
            potential_dur = t["end"] - prev["start"]

            if (
                t["cluster_id"] == prev["cluster_id"]
                and 0.0 <= gap <= self.merge_gap_threshold
                and potential_dur <= self.max_chunk_duration
            ):
                prev["end"] = t["end"]
            else:
                merged.append(dict(t))

        # Add acoustic padding and format chunks
        max_limit = total_audio_duration or (end_time if end_time else 999999.0)
        chunks: List[Dict[str, Any]] = []
        for i, m in enumerate(merged):
            dur = m["end"] - m["start"]
            if dur < self.min_chunk_duration:
                # Still keep if >= 0.3s
                if dur < 0.3:
                    continue

            p_start = max(0.0, round(m["start"] - self.pad_onset, 3))
            p_end = min(max_limit, round(m["end"] + self.pad_offset, 3))

            chunks.append({
                "chunk_id": i,
                "start": m["start"],
                "end": m["end"],
                "pad_start": p_start,
                "pad_end": p_end,
                "duration": round(dur, 3),
                "cluster_id": m["cluster_id"],
            })

        return chunks
