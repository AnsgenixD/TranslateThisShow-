"""
aligner.py - Millisecond-Accurate Forced Alignment & Acoustic Boundary Snapping.

Uses MLX-native `Qwen3-ForcedAligner-0.6B-8bit` to produce character-level timestamps,
followed by acoustic RMS/VAD boundary snapping to guarantee tight sync with mouth movement.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf


class ForcedAligner:
    """
    Apple Silicon MLX Character-Level Forced Aligner with Acoustic Energy Snapping.
    """

    def __init__(self, model_repo: str = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"):
        self.model_repo = model_repo
        self.model = None
        self._load_failed = False

    def _ensure_loaded(self) -> bool:
        if self.model is None and not self._load_failed:
            try:
                from mlx_audio.stt import load
                print(f"🎙️ Loading Forced Aligner: {self.model_repo}...")
                self.model = load(self.model_repo)
                print("✅ Forced Aligner ready.")
                return True
            except ImportError:
                print("\n💡 [Optional Feature] Millisecond Forced Alignment requires 'mlx-audio'.")
                print("   To enable character-level CTC forced alignment on Apple Silicon:")
                print("     pip install mlx-audio")
                print(f"   Model: {self.model_repo}")
                print("   Using linear timestamp interpolation fallback...\n")
                self._load_failed = True
                return False
            except Exception as e:
                print(f"\n💡 [Notice] Forced Aligner '{self.model_repo}' not available ({e}).")
                print("   Using linear timestamp interpolation fallback...\n")
                self._load_failed = True
                return False
        return self.model is not None

    def unload(self) -> None:
        """Frees model from unified memory."""
        if self.model is not None:
            del self.model
            self.model = None
            import mlx.core as mx
            mx.clear_cache()

    def align_text(
        self,
        audio_16k: np.ndarray,
        text: str,
        language: str = "Chinese",
        base_offset_sec: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Aligns a text string against an audio array.
        Returns a list of character tokens with start and end timestamps in seconds.
        """
        clean_text = re.sub(r"[\s\r\n\t]", "", text)
        if not clean_text or len(audio_16k) < 1600:
            return []

        if not self._ensure_loaded():
            total_dur = len(audio_16k) / 16000.0
            char_dur = total_dur / max(1, len(clean_text))
            return [
                {
                    "char": ch,
                    "start": round(base_offset_sec + i * char_dur, 3),
                    "end": round(base_offset_sec + (i + 1) * char_dur, 3),
                }
                for i, ch in enumerate(clean_text)
            ]

        # Write temporary wav for aligner input
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_16k, 16000)
            try:
                res = self.model.generate(
                    audio=tmp.name,
                    text=clean_text,
                    language=language,
                )
            except Exception as e:
                print(f"⚠️ Aligner error for text '{clean_text[:20]}': {e}")
                return []

        char_items: List[Dict[str, Any]] = []
        for item in res:
            char_items.append({
                "char": item.text,
                "start": round(base_offset_sec + float(item.start_time), 3),
                "end": round(base_offset_sec + float(item.end_time), 3),
            })

        return char_items

    @staticmethod
    def snap_boundary_to_acoustics(
        audio_16k: np.ndarray,
        nominal_start: float,
        nominal_end: float,
        search_window: float = 0.20,
        energy_threshold_ratio: float = 0.25,
        lead_in: float = 0.06,
        tail_release: float = 0.12,
    ) -> Tuple[float, float]:
        """
        Refines nominal start and end times by snapping to real vocal energy transitions.
        Adds broadcast lead-in (-60ms) and tail release (+120ms).
        """
        sr = 16000
        total_dur = len(audio_16k) / sr

        # 1. Start boundary search
        s_center = int(nominal_start * sr)
        w_samp = int(search_window * sr)
        s_left = max(0, s_center - w_samp)
        s_right = min(len(audio_16k), s_center + w_samp)

        snapped_start = nominal_start
        if s_right > s_left + 160:
            clip = audio_16k[s_left:s_right]
            frame_size = 160  # 10ms
            energies = [
                float(np.sqrt(np.mean(clip[i:i + frame_size] ** 2)))
                for i in range(0, len(clip) - frame_size, frame_size)
            ]
            if energies:
                max_e = max(energies)
                thresh = max_e * energy_threshold_ratio
                # Find first frame crossing threshold
                for fi, e in enumerate(energies):
                    if e >= thresh:
                        snapped_start = s_left / sr + (fi * 10 / 1000.0)
                        break

        # 2. End boundary search
        e_center = int(nominal_end * sr)
        e_left = max(0, e_center - w_samp)
        e_right = min(len(audio_16k), e_center + w_samp)

        snapped_end = nominal_end
        if e_right > e_left + 160:
            clip = audio_16k[e_left:e_right]
            frame_size = 160
            energies = [
                float(np.sqrt(np.mean(clip[i:i + frame_size] ** 2)))
                for i in range(0, len(clip) - frame_size, frame_size)
            ]
            if energies:
                max_e = max(energies)
                thresh = max_e * energy_threshold_ratio
                # Find last frame crossing threshold (from right to left)
                for fi in range(len(energies) - 1, -1, -1):
                    if energies[fi] >= thresh:
                        snapped_end = e_left / sr + ((fi + 1) * 10 / 1000.0)
                        break

        # Apply broadcast padding
        final_start = max(0.0, round(snapped_start - lead_in, 3))
        final_end = min(total_dur, round(snapped_end + tail_release, 3))
        if final_end <= final_start + 0.15:
            final_end = round(final_start + 0.35, 3)

        return final_start, final_end
