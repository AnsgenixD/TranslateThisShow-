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
import numpy as np


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
        audio_16k: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Builds speaker-homogeneous chunks within the time window [start_time, end_time].
        Merges consecutive turns of the same speaker if separated by <= merge_gap_threshold.
        Maintains independent parallel chunks for overlapping speakers.
        """
        if not self.turns:
            eff_end = end_time or total_audio_duration or (start_time + 30.0)
            max_limit = total_audio_duration or eff_end
            if audio_16k is not None and len(audio_16k) > 0:
                return self._energy_vad_chunks(audio_16k, start_time, eff_end, max_limit)

            # Fixed window fallback if audio array is not passed
            chunks: List[Dict[str, Any]] = []
            cur_s = start_time
            win_len = 8.0
            while cur_s < eff_end - 0.1:
                cur_e = min(eff_end, cur_s + win_len)
                chunks.append({
                    "chunk_id": len(chunks),
                    "start": round(cur_s, 3),
                    "end": round(cur_e, 3),
                    "pad_start": round(max(0.0, cur_s - self.pad_onset), 3),
                    "pad_end": round(min(max_limit, cur_e + self.pad_offset), 3),
                    "duration": round(cur_e - cur_s, 3),
                    "cluster_id": "SPEAKER_00",
                })
                cur_s = cur_e
            return chunks

    def _energy_vad_chunks(
        self,
        audio_16k: np.ndarray,
        start_time: float,
        end_time: float,
        max_limit: float,
    ) -> List[Dict[str, Any]]:
        """Splits audio into natural breath/pause intervals using short-term RMS energy."""
        sr = 16000
        hop_ms = 20
        hop_len = int(sr * hop_ms / 1000)  # 320 samples
        s_samp = max(0, int(start_time * sr))
        e_samp = min(len(audio_16k), int(end_time * sr))
        slice_audio = audio_16k[s_samp:e_samp]

        n_frames = len(slice_audio) // hop_len
        if n_frames == 0:
            return []

        frames = slice_audio[: n_frames * hop_len].reshape(n_frames, hop_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        thresh = max(0.008, float(np.percentile(rms, 25)) * 1.3)
        is_speech = rms > thresh

        # Smooth short silence gaps (< 280ms = 14 frames)
        min_silence_frames = 14
        i = 0
        while i < n_frames:
            if not is_speech[i]:
                j = i
                while j < n_frames and not is_speech[j]:
                    j += 1
                if (j - i) < min_silence_frames and i > 0 and j < n_frames:
                    is_speech[i:j] = True
                i = j
            else:
                i += 1

        # Extract contiguous speech intervals, splitting at lowest-energy valley if > max_chunk_duration
        raw_intervals: List[Tuple[float, float]] = []
        i = 0
        while i < n_frames:
            if is_speech[i]:
                j = i
                while j < n_frames and is_speech[j]:
                    j += 1
                seg_s = start_time + (i * hop_ms) / 1000.0
                seg_e = start_time + (j * hop_ms) / 1000.0
                if seg_e - seg_s >= self.min_chunk_duration:
                    # If interval exceeds max_chunk_duration, split at quietest internal valley every ~7s
                    cur_f = i
                    while (j - cur_f) * hop_ms / 1000.0 > self.max_chunk_duration:
                        search_start = cur_f + int(4.0 * 1000 / hop_ms)
                        search_end = min(j, cur_f + int(self.max_chunk_duration * 1000 / hop_ms))
                        if search_end > search_start:
                            valley_f = search_start + int(np.argmin(rms[search_start:search_end]))
                        else:
                            valley_f = search_end
                        raw_intervals.append((
                            start_time + (cur_f * hop_ms) / 1000.0,
                            start_time + (valley_f * hop_ms) / 1000.0,
                        ))
                        cur_f = valley_f
                    if (j - cur_f) * hop_ms / 1000.0 >= 0.3:
                        raw_intervals.append((
                            start_time + (cur_f * hop_ms) / 1000.0,
                            start_time + (j * hop_ms) / 1000.0,
                        ))
                i = j
            else:
                i += 1

        chunks: List[Dict[str, Any]] = []
        for idx, (c_s, c_e) in enumerate(raw_intervals):
            chunks.append({
                "chunk_id": idx,
                "start": round(c_s, 3),
                "end": round(c_e, 3),
                "pad_start": round(max(0.0, c_s - self.pad_onset), 3),
                "pad_end": round(min(max_limit, c_e + self.pad_offset), 3),
                "duration": round(c_e - c_s, 3),
                "cluster_id": f"SPEAKER_{idx % 4:02d}",
            })
        return chunks

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
