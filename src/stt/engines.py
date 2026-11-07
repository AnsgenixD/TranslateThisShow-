"""
engines.py - Unified ASR Engine Adapters for Apple Silicon Metal / MLX.

Provides a unified interface across:
1. MLX Whisper (whisper-large-v3-turbo / whisper-large-v3-mlx)
2. Qwen3-ASR (0.6B / 1.7B 8-bit)
3. Fun-ASR-Nano (0.8B)
4. SenseVoice-Small (234M rich event ASR)
5. FireRedASR2-AED (1.18B Conformer-AED)
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import mlx.core as mx
import numpy as np
import soundfile as sf


class BaseASREngine(ABC):
    """Abstract base adapter for MLX speech recognition engines."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._fallback = None

    @abstractmethod
    def load(self) -> None:
        pass

    def unload(self) -> None:
        """Unloads weights from unified memory."""
        if self.model is not None:
            del self.model
            self.model = None
            mx.clear_cache()

    def _chunk_and_transcribe(
        self,
        audio_16k: np.ndarray,
        transcribe_fn,
        chunk_sec: float = 20.0,
        overlap_sec: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Slices long audio into manageable chunks if longer than chunk_sec + 5s."""
        total_sec = len(audio_16k) / 16000.0
        if total_sec <= chunk_sec + 5.0:
            return transcribe_fn(audio_16k, 0.0)

        results = []
        step_sec = chunk_sec - overlap_sec
        curr_s = 0.0
        while curr_s < total_sec:
            curr_e = min(total_sec, curr_s + chunk_sec)
            s_samp = int(curr_s * 16000)
            e_samp = int(curr_e * 16000)
            clip = audio_16k[s_samp:e_samp]
            if len(clip) >= 1600:
                sub_res = transcribe_fn(clip, curr_s)
                results.extend(sub_res)
            curr_s += step_sec
        return results

    @abstractmethod
    def transcribe(
        self,
        audio_16k: np.ndarray,
        hotwords: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Transcribes a 16kHz mono audio slice.
        Returns a list of segment dicts with keys: 'start', 'end', 'text', 'confidence', 'events'.
        """
        pass


class WhisperMLXEngine(BaseASREngine):
    """MLX Whisper adapter (whisper-large-v3-turbo / whisper-large-v3-mlx)."""

    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-turbo"):
        super().__init__(model_name)

    def load(self) -> None:
        # mlx_whisper loads weights on demand or caches in memory
        pass

    def transcribe(
        self,
        audio_16k: np.ndarray,
        hotwords: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        import mlx_whisper

        prompt = system_prompt or "以下是普通话对话，请使用标准简体中文标点符号。"
        if hotwords:
            prompt = f"词汇：{'、'.join(hotwords[:12])}。{prompt}"

        res = mlx_whisper.transcribe(
            audio_16k,
            path_or_hf_repo=self.model_name,
            language="zh",
            condition_on_previous_text=False,
            word_timestamps=False,
            hallucination_silence_threshold=1.0,
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.6,
            initial_prompt=prompt[:220],
        )

        segments = []
        for s in res.get("segments", []):
            txt = (s.get("text") or "").strip()
            if txt:
                segments.append({
                    "start": round(float(s.get("start", 0.0)), 3),
                    "end": round(float(s.get("end", 0.0)), 3),
                    "text": txt,
                    "avg_logprob": round(float(s.get("avg_logprob", 0.0)), 3),
                    "events": ["Speech"],
                })
        return segments


class Qwen3ASREngine(BaseASREngine):
    """Qwen3-ASR adapter (0.6B / 1.7B 8-bit)."""

    def __init__(self, model_name: str = "mlx-community/Qwen3-ASR-0.6B-8bit"):
        super().__init__(model_name)

    def load(self) -> None:
        if self.model is None and self._fallback is None:
            try:
                from mlx_audio.stt import load
                print(f"🎙️ Loading Qwen3-ASR: {self.model_name}...")
                self.model = load(self.model_name)
                print("✅ Qwen3-ASR loaded.")
            except ImportError:
                print("\n💡 [Optional Engine] Qwen3-ASR requires 'mlx-audio'.")
                print("   To install Qwen3-ASR support for Apple Silicon:")
                print("     pip install mlx-audio")
                print("   Supported models:")
                print("     - mlx-community/Qwen3-ASR-0.6B-8bit (Fast, ~1GB VRAM, recommended)")
                print("     - mlx-community/Qwen3-ASR-1.7B-8bit (Highest accuracy, ~2.5GB VRAM)")
                print("   Falling back to Whisper Large-v3 Turbo...\n")
                self._fallback = WhisperMLXEngine()
                self._fallback.load()
            except Exception as e:
                print(f"\n⚠️ Could not load Qwen3-ASR model '{self.model_name}': {e}")
                print("   Falling back to Whisper Large-v3 Turbo...\n")
                self._fallback = WhisperMLXEngine()
                self._fallback.load()

    def transcribe(
        self,
        audio_16k: np.ndarray,
        hotwords: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.load()
        if self._fallback is not None:
            return self._fallback.transcribe(audio_16k, hotwords=hotwords, system_prompt=system_prompt)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_16k, 16000)
            try:
                res = self.model.generate(
                    tmp.name,
                    language="Chinese",
                    hotwords=hotwords or None,
                    system_prompt=system_prompt or None,
                )
            except Exception as e:
                print(f"⚠️ Qwen3-ASR transcription error: {e}")
                return []

        text = (res.text or "").strip()
        if not text:
            return []

        segments = []
        if hasattr(res, "segments") and res.segments:
            for s in res.segments:
                txt = (s.get("text") or "").strip()
                if txt:
                    segments.append({
                        "start": round(float(s.get("start", 0.0)), 3),
                        "end": round(float(s.get("end", 0.0)), 3),
                        "text": txt,
                        "events": ["Speech"],
                    })
        else:
            dur = len(audio_16k) / 16000.0
            segments.append({
                "start": 0.0,
                "end": round(dur, 3),
                "text": text,
                "events": ["Speech"],
            })

        return segments


class FunASRNanoEngine(BaseASREngine):
    """Fun-ASR-Nano adapter (0.8B)."""

    def __init__(self, model_name: str = "mlx-community/Fun-ASR-Nano-2512"):
        super().__init__(model_name)

    def load(self) -> None:
        if self.model is None and self._fallback is None:
            try:
                from mlx_audio.stt import load
                print(f"🎙️ Loading Fun-ASR-Nano: {self.model_name}...")
                self.model = load(self.model_name)
                print("✅ Fun-ASR-Nano loaded.")
            except ImportError:
                print("\n💡 [Optional Engine] Fun-ASR-Nano requires 'mlx-audio'.")
                print("   To install: pip install mlx-audio")
                print("   Falling back to Whisper Large-v3 Turbo...\n")
                self._fallback = WhisperMLXEngine()
                self._fallback.load()
            except Exception as e:
                print(f"\n⚠️ Could not load Fun-ASR-Nano model '{self.model_name}': {e}")
                print("   Falling back to Whisper Large-v3 Turbo...\n")
                self._fallback = WhisperMLXEngine()
                self._fallback.load()

    def transcribe(
        self,
        audio_16k: np.ndarray,
        hotwords: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.load()
        if self._fallback is not None:
            return self._fallback.transcribe(audio_16k, hotwords=hotwords, system_prompt=system_prompt)

        def _transcribe_slice(clip: np.ndarray, offset_sec: float) -> List[Dict[str, Any]]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, clip, 16000)
                try:
                    res = self.model.generate(
                        tmp.name,
                        language="zh",
                        hotwords=hotwords or None,
                    )
                except Exception as e:
                    print(f"⚠️ Fun-ASR-Nano transcription error: {e}")
                    return []

            text = (res.text or "").strip()
            if not text:
                return []
            dur = len(clip) / 16000.0
            return [{
                "start": round(offset_sec, 3),
                "end": round(offset_sec + dur, 3),
                "text": text,
                "events": ["Speech"],
            }]

        return self._chunk_and_transcribe(audio_16k, _transcribe_slice, chunk_sec=20.0)


class SenseVoiceEngine(BaseASREngine):
    """SenseVoice-Small adapter (rich events: laughter, BGM, applause, emotion)."""

    def __init__(self, model_name: str = "mlx-community/SenseVoiceSmall"):
        super().__init__(model_name)

    def load(self) -> None:
        if self.model is None and self._fallback is None:
            try:
                from mlx_audio.stt import load
                print(f"🎙️ Loading SenseVoiceSmall: {self.model_name}...")
                self.model = load(self.model_name)
                print("✅ SenseVoiceSmall loaded.")
            except ImportError:
                print("\n💡 [Optional Engine] SenseVoice-Small requires 'mlx-audio'.")
                print("   To install: pip install mlx-audio")
                print("   SenseVoice detects emotions, laughter, applause and speech events at 45x realtime.")
                print("   Falling back to Whisper Large-v3 Turbo...\n")
                self._fallback = WhisperMLXEngine()
                self._fallback.load()
            except Exception as e:
                print(f"\n⚠️ Could not load SenseVoice model '{self.model_name}': {e}")
                print("   Falling back to Whisper Large-v3 Turbo...\n")
                self._fallback = WhisperMLXEngine()
                self._fallback.load()

    def transcribe(
        self,
        audio_16k: np.ndarray,
        hotwords: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.load()
        if self._fallback is not None:
            return self._fallback.transcribe(audio_16k, hotwords=hotwords, system_prompt=system_prompt)

        def _transcribe_slice(clip: np.ndarray, offset_sec: float) -> List[Dict[str, Any]]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, clip, 16000)
                try:
                    res = self.model.generate(tmp.name, language="zh", use_itn=True)
                except Exception as e:
                    print(f"⚠️ SenseVoice transcription error: {e}")
                    return []

            text = (res.text or "").strip()
            dur = len(clip) / 16000.0
            events = []
            emotion = "neutral"
            if hasattr(res, "segments") and res.segments:
                seg0 = res.segments[0]
                if seg0.get("event"):
                    events.append(seg0["event"])
                if seg0.get("emotion"):
                    emotion = seg0["emotion"]

            if not text:
                return []
            return [{
                "start": round(offset_sec, 3),
                "end": round(offset_sec + dur, 3),
                "text": text,
                "emotion": emotion,
                "events": events or ["Speech"],
            }]

        return self._chunk_and_transcribe(audio_16k, _transcribe_slice, chunk_sec=20.0)


class FireRedASREngine(BaseASREngine):
    """FireRedASR2-AED adapter."""

    def __init__(self, model_name: str = "mlx-community/FireRedASR2-AED-mlx"):
        super().__init__(model_name)

    def load(self) -> None:
        if self.model is None:
            from mlx_audio.stt import load
            print(f"🎙️ Loading FireRedASR2-AED: {self.model_name}...")
            self.model = load(self.model_name)
            print("✅ FireRedASR2-AED loaded.")

    def transcribe(
        self,
        audio_16k: np.ndarray,
        hotwords: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.load()

        def _transcribe_slice(clip: np.ndarray, offset_sec: float) -> List[Dict[str, Any]]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, clip, 16000)
                try:
                    res = self.model.generate(tmp.name)
                except Exception as e:
                    print(f"⚠️ FireRedASR2 transcription error: {e}")
                    return []

            text = (res.text or "").strip()
            if not text:
                return []
            dur = len(clip) / 16000.0
            return [{
                "start": round(offset_sec, 3),
                "end": round(offset_sec + dur, 3),
                "text": text,
                "events": ["Speech"],
            }]

        return self._chunk_and_transcribe(audio_16k, _transcribe_slice, chunk_sec=20.0)


def get_engine(engine_name: str) -> BaseASREngine:
    """Factory creating the appropriate ASR engine instance."""
    engine_name = engine_name.lower()
    if "whisper" in engine_name:
        if "large-v3-mlx" in engine_name:
            return WhisperMLXEngine("mlx-community/whisper-large-v3-mlx")
        return WhisperMLXEngine("mlx-community/whisper-large-v3-turbo")
    elif "qwen3-1.7b" in engine_name or "qwen3_1.7b" in engine_name:
        return Qwen3ASREngine("mlx-community/Qwen3-ASR-1.7B-8bit")
    elif "qwen3" in engine_name:
        return Qwen3ASREngine("mlx-community/Qwen3-ASR-0.6B-8bit")
    elif "fun" in engine_name or "nano" in engine_name:
        return FunASRNanoEngine("mlx-community/Fun-ASR-Nano-2512")
    elif "sense" in engine_name:
        return SenseVoiceEngine("mlx-community/SenseVoiceSmall")
    elif "firered" in engine_name:
        return FireRedASREngine("mlx-community/FireRedASR2-AED-mlx")
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
