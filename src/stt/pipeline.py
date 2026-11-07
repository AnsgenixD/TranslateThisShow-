"""
pipeline.py - Production Broadcast-Grade STT & Subtitling Pipeline.

Coordinates:
- Context & Autonomous Roster Management (context.py)
- Diarization-Aware Overlap-Preserving Audio Chunking (chunker.py)
- Frontier Apple Silicon MLX ASR Engines (engines.py)
- Phonetic Pinyin Homophone Correction (name_fixer.py)
- Character-Level Forced Alignment & Boundary Snapping (aligner.py)
- Semantic Subtitle & Dubbing Unit Segmentation (segmenter.py)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import mlx.core as mx
import numpy as np
import soundfile as sf

try:
    from .aligner import ForcedAligner
    from .chunker import DiarizationChunker
    from .context import STTContextManager
    from .engines import BaseASREngine, get_engine
    from .name_fixer import PhoneticNameFixer
    from .segmenter import SemanticSegmenter
except ImportError:
    from CleanProject.src.stt.aligner import ForcedAligner
    from CleanProject.src.stt.chunker import DiarizationChunker
    from CleanProject.src.stt.context import STTContextManager
    from CleanProject.src.stt.engines import BaseASREngine, get_engine
    from CleanProject.src.stt.name_fixer import PhoneticNameFixer
    from CleanProject.src.stt.segmenter import SemanticSegmenter


def _load_audio_16k(audio_path: str) -> np.ndarray:
    """Loads any audio file as 16kHz mono float32 numpy array."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


class BroadcastSTTPipeline:
    """
    End-to-End Broadcast STT Pipeline for Chinese Reality TV.
    """

    def __init__(
        self,
        primary_engine: str = "qwen3",
        enable_forced_alignment: bool = True,
        enable_name_fixer: bool = True,
        enable_sensevoice_events: bool = False,
        dossier_path: str = "output/show_dossier.json",
        roster_path: str = "output/face_roster/confirmed_actor_roster.json",
        turns_path: str = "output/diar_v2/pyannote_turns.json",
    ):
        self.primary_engine_name = primary_engine
        self.enable_forced_alignment = enable_forced_alignment
        self.enable_name_fixer = enable_name_fixer
        self.enable_sensevoice_events = enable_sensevoice_events

        self.context_mgr = STTContextManager(dossier_path=dossier_path, roster_path=roster_path)
        self.chunker = DiarizationChunker(turns_path=turns_path)
        self.segmenter = SemanticSegmenter()

        # Build name fixer from current context
        all_names = self.context_mgr.get_all_names()
        all_terms = self.context_mgr.get_all_terms()
        self.name_fixer = PhoneticNameFixer(names=all_names, terms=all_terms)

        self.aligner: Optional[ForcedAligner] = None
        if self.enable_forced_alignment:
            self.aligner = ForcedAligner()

    def process(
        self,
        audio_path: str,
        orig_audio_path: Optional[str] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline on the input audio over [start_time, end_time].
        Returns a dict containing:
        - 'subtitles': list of broadcast subtitle cues
        - 'dubbing_units': list of sentence-complete TTS dubbing units
        - 'raw_segments': list of transcribed speaker chunks
        - 'character_alignment': list of character-level timestamps
        - 'stats': execution metrics (duration, RTF, memory)
        """
        t0 = time.time()
        print(f"\n🚀 Running Broadcast STT Pipeline [{self.primary_engine_name}]...")

        # 1. Load Audio
        audio_16k = _load_audio_16k(audio_path)
        total_audio_dur = len(audio_16k) / 16000.0
        eff_end = min(total_audio_dur, end_time) if end_time else total_audio_dur
        eff_dur = eff_end - start_time
        print(f"🎵 Audio loaded: {total_audio_dur:.1f}s total | Window: {start_time:.1f}s -> {eff_end:.1f}s ({eff_dur:.1f}s)")

        # 2. Build Diarization Chunks
        chunks = self.chunker.get_chunks(
            start_time=start_time,
            end_time=eff_end,
            total_audio_duration=total_audio_dur,
        )
        print(f"🧩 Diarization-aware chunking: generated {len(chunks)} speaker turns.")

        # 3. Load Primary ASR Engine
        engine: BaseASREngine = get_engine(self.primary_engine_name)
        hotwords = self.context_mgr.get_hotwords()
        sys_prompt = self.context_mgr.get_system_prompt()
        print(f"🎯 Roster Biasing: {len(hotwords)} hotwords active ({', '.join(hotwords[:6])}...)")

        raw_chunk_results: List[Dict[str, Any]] = []
        n_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            c_start = chunk["pad_start"]
            c_end = chunk["pad_end"]
            cluster = chunk["cluster_id"]

            s_samp = int(c_start * 16000)
            e_samp = int(c_end * 16000)
            clip = audio_16k[s_samp:e_samp]
            if len(clip) < 1600:
                continue

            # Energy check: skip silence
            rms = float(np.sqrt(np.mean(clip ** 2)))
            if rms < 0.008:
                continue

            # Pass hotwords adaptively: shorter chunks get only primary cast names to prevent prompt dumping
            chunk_dur = len(clip) / 16000.0
            chunk_hotwords = hotwords[:8] if chunk_dur >= 1.2 else None
            chunk_prompt = sys_prompt if chunk_dur >= 2.0 else None

            # Transcribe clip
            clip_res = engine.transcribe(
                clip,
                hotwords=chunk_hotwords,
                system_prompt=chunk_prompt,
            )

            hw_set = set(hotwords)
            for item in clip_res:
                txt = item.get("text", "").strip()
                if not txt:
                    continue

                # Anti-hallucination: detect prompt dumps (comma-separated list of hotwords)
                parts = [p.strip() for p in re.split(r"[，,、\s]+", txt) if p.strip()]
                if len(parts) >= 3:
                    matches = sum(1 for p in parts if p in hw_set)
                    if matches / len(parts) >= 0.65:
                        continue

                # Local clip timestamps -> absolute timestamps
                rel_s = item.get("start", 0.0)
                rel_e = item.get("end", c_end - c_start)
                abs_s = round(c_start + rel_s, 3)
                abs_e = round(c_start + rel_e, 3)

                raw_chunk_results.append({
                    "chunk_id": chunk["chunk_id"],
                    "start": abs_s,
                    "end": abs_e,
                    "text": txt,
                    "cluster_id": cluster,
                    "rms": round(rms, 4),
                    "events": item.get("events", ["Speech"]),
                })

            if on_progress:
                on_progress(idx + 1, n_chunks, f"Chunk {idx+1}/{n_chunks} [{cluster}]")

        # Unload primary ASR to free unified memory for aligner
        engine.unload()
        mx.clear_cache()
        print(f"📝 Raw chunks transcribed: {len(raw_chunk_results)} lines.")

        # 4. Phonetic Homophone Correction (Pinyin Fuzzy Matching)
        if self.enable_name_fixer:
            if not self.context_mgr.get_all_names() and raw_chunk_results:
                self.context_mgr.discover_entities_with_llm([s["text"] for s in raw_chunk_results])
                self.name_fixer = PhoneticNameFixer(
                    names=self.context_mgr.get_all_names(),
                    terms=self.context_mgr.get_all_terms(),
                )
            total_fixes = 0
            for seg in raw_chunk_results:
                fixed, reps = self.name_fixer.fix_text(seg["text"])
                if reps:
                    total_fixes += len(reps)
                    seg["name_fixes"] = reps
                seg["text"] = fixed
            if total_fixes > 0:
                print(f"✨ Phonetic Name Fixer: resolved {total_fixes} homophone/entity errors.")

        # 5. Forced Alignment & Acoustic Boundary Snapping
        char_alignments: List[Dict[str, Any]] = []
        if self.enable_forced_alignment and self.aligner is not None:
            print("⏱️ Running Qwen3 Forced Alignment & Acoustic Snapping...")
            for seg in raw_chunk_results:
                s_samp = max(0, int((seg["start"] - 0.2) * 16000))
                e_samp = min(len(audio_16k), int((seg["end"] + 0.2) * 16000))
                c_audio = audio_16k[s_samp:e_samp]
                base_sec = s_samp / 16000.0

                aligned_chars = self.aligner.align_text(
                    audio_16k=c_audio,
                    text=seg["text"],
                    language="Chinese",
                    base_offset_sec=base_sec,
                )

                if aligned_chars:
                    char_alignments.extend(aligned_chars)
                    # Refine segment start and end by first/last character timestamps + acoustic snapping
                    first_c = aligned_chars[0]["start"]
                    last_c = aligned_chars[-1]["end"]
                    snapped_s, snapped_e = self.aligner.snap_boundary_to_acoustics(
                        audio_16k=audio_16k,
                        nominal_start=first_c,
                        nominal_end=last_c,
                    )
                    seg["start"] = snapped_s
                    seg["end"] = snapped_e

            self.aligner.unload()
            mx.clear_cache()

        # 6. Semantic Segmentation for Subtitles & Dubbing Units
        subtitles = self.segmenter.build_subtitles(raw_chunk_results)
        dubbing_units = self.segmenter.build_dubbing_units(raw_chunk_results)

        total_time = time.time() - t0
        rtf = total_time / max(eff_dur, 0.1)
        print(f"🎉 Pipeline complete in {total_time:.2f}s (RTF: {rtf:.3f}x realtime)!")
        print(f"   Subtitles generated: {len(subtitles)} cues | Dubbing units: {len(dubbing_units)}")

        return {
            "subtitles": subtitles,
            "dubbing_units": dubbing_units,
            "raw_segments": raw_chunk_results,
            "character_alignment": char_alignments,
            "stats": {
                "wall_time_sec": round(total_time, 2),
                "audio_duration_sec": round(eff_dur, 2),
                "real_time_factor": round(rtf, 3),
                "num_subtitles": len(subtitles),
                "num_dubbing_units": len(dubbing_units),
                "engine": self.primary_engine_name,
            },
        }

    def export_srt(self, subtitles: List[Dict[str, Any]], output_srt_path: str) -> None:
        """Exports subtitles to standard SRT file with speaker tags."""
        out_p = Path(output_srt_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for cue in subtitles:
            idx = cue.get("index", 1)
            st = cue["start"]
            et = cue["end"]
            txt = cue["text"]
            spk = cue.get("cluster_id", "")

            # Format SRT timestamp: HH:MM:SS,mmm
            s_h = int(st // 3600)
            s_m = int((st % 3600) // 60)
            s_s = int(st % 60)
            s_ms = int(round((st - int(st)) * 1000))

            e_h = int(et // 3600)
            e_m = int((et % 3600) // 60)
            e_s = int(et % 60)
            e_ms = int(round((et - int(et)) * 1000))

            s_str = f"{s_h:02d}:{s_m:02d}:{s_s:02d},{s_ms:03d}"
            e_str = f"{e_h:02d}:{e_m:02d}:{e_s:02d},{e_ms:03d}"

            lines.append(str(idx))
            lines.append(f"{s_str} --> {e_str}")
            # Format text (preserve any multi-line split)
            lines.append(txt)
            lines.append("")

        out_p.write_text("\n".join(lines), encoding="utf-8")
        print(f"💾 Exported SRT -> {output_srt_path} ({len(subtitles)} cues)")
