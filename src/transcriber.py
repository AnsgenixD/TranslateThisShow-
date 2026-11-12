"""
Adaptive Speech-to-Text Subtitler (Apple Silicon MLX).

Features:
  1. Uses `mlx-community/whisper-large-v3-turbo` by default (full 128-mel Large-v3 encoder,
     4x faster turbo decoder) with anti-hallucination decoding (`condition_on_previous_text=False`,
     `word_timestamps=True`, `hallucination_silence_threshold=1.0`, `compression_ratio_threshold=2.0`).
  2. Acoustic RMS energy gating + consecutive repetition deduplication (prevents BGM/theme-song
     loops like 44x "有" / "Yes").
  3. Strict non-overlapping timestamp enforcement + smart clause merging so micro-captions
     become complete, natural sentences for both reading and TTS dubbing.
  4. `resubtitle_segments()`: Targeted re-transcription of segments flagged as "off" by DeepSeek,
     escalating to full `mlx-community/whisper-large-v3-mlx` conditioned on DeepSeek's contextual
     dossier and scene hints.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import librosa
import numpy as np
import soundfile as sf
import mlx_whisper

DEFAULT_PRIMARY_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_ESCALATION_MODEL = "mlx-community/whisper-large-v3-mlx"

_TERMINAL_PUNCT = set("。？！?!…")
_SHORT_FILLERS = {"有", "有啊", "可以", "好", "好的", "对", "对的", "嗯", "啊", "哦", "是", "是的"}
_WHISPER_BOILERPLATE = (
    "点赞", "订阅", "转发", "打赏", "明镜", "点点栏目", "字幕志愿者",
    "字幕由", "谢谢观看", "感谢观看", "Amara", "Subtitles by", "优优独播",
)


def _is_boilerplate_hallucination(text: str, duration: float) -> bool:
    """Detects classic Whisper training-set watermark hallucinations during music/intro segments."""
    for kw in _WHISPER_BOILERPLATE:
        if kw in text:
            return True
    # A 10+ second segment with < 1.2 characters/sec is music/noise hallucination
    clean_chars = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    if duration >= 8.0 and len(clean_chars) / max(duration, 1.0) < 1.2:
        return True
    return False


def _load_audio_16k(audio_path: str) -> np.ndarray:
    """Loads any audio file as 16kHz mono float32 numpy array."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        if sr == 44100:
            audio = resample_poly(audio, 160, 441)
        elif sr == 48000:
            audio = resample_poly(audio, 1, 3)
        elif sr == 24000:
            audio = resample_poly(audio, 2, 3)
        else:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def _clean_and_gate_segments(
    raw_segments: List[dict],
    audio_16k: np.ndarray,
    min_rms: float = 0.006,
    max_no_speech_prob: float = 0.65,
    max_compression_ratio: float = 2.2,
) -> List[dict]:
    """
    Filters out BGM hallucinations, silent segments, and repetition loops,
    and guarantees strictly non-overlapping timestamps.
    """
    cleaned: List[dict] = []
    repeat_streak = 0
    last_norm_text = ""

    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        start = max(0.0, float(seg.get("start", 0.0)))
        end = float(seg.get("end", start))
        if end <= start + 0.08:
            continue

        # 1. Check Whisper internal confidence metrics if present
        no_speech = float(seg.get("no_speech_prob", 0.0))
        comp_ratio = float(seg.get("compression_ratio", 1.0))
        if no_speech > max_no_speech_prob:
            continue
        if comp_ratio > max_compression_ratio and len(set(text)) <= 3:
            continue

        # 2. Check actual acoustic RMS energy in the audio slice
        s_idx = int(start * 16000)
        e_idx = min(len(audio_16k), int(end * 16000))
        if e_idx <= s_idx:
            continue
        slice_audio = audio_16k[s_idx:e_idx]
        rms = float(np.sqrt(np.mean(slice_audio ** 2)))
        if rms < min_rms:
            continue

        # 3. Collapse internal character stuttering (e.g. "有有有有" -> "有")
        text = re.sub(r"(.{1,2})\1{3,}", r"\1", text).strip()
        if _is_boilerplate_hallucination(text, end - start):
            continue

        # 4. Detect consecutive identical filler loops (e.g. 44x "有" / "可以" during music)
        norm_text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        if norm_text == last_norm_text and (norm_text in _SHORT_FILLERS or len(norm_text) <= 2):
            repeat_streak += 1
            if repeat_streak >= 2:
                # Drop 3rd+ consecutive identical micro-filler
                continue
        else:
            repeat_streak = 0
            last_norm_text = norm_text

        # 5. Enforce non-overlapping timestamps with previous segment
        if cleaned and start < cleaned[-1]["end"]:
            if end <= cleaned[-1]["end"] + 0.15:
                # Completely subsumed by previous segment
                continue
            start = round(cleaned[-1]["end"] + 0.01, 3)

        cleaned.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "rms": round(rms, 4),
            "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 3),
        })

    return cleaned


def merge_segments(
    segments: List[dict],
    max_gap: float = 0.65,
    max_duration: float = 6.5,
    max_chars: int = 42,
) -> List[dict]:
    """
    Merges short consecutive subtitle clauses separated by tiny pauses (<= max_gap)
    into complete, natural sentences. Eliminates sub-second flickering captions and
    prevents overlapping English TTS clips during dubbing.
    """
    if not segments:
        return []

    merged: List[dict] = []
    current: Optional[dict] = None

    for s in segments:
        text = s.get("text", "").strip()
        if not text:
            continue

        if current is None:
            current = dict(s)
            continue

        gap = s["start"] - current["end"]
        duration = s["end"] - current["start"]
        combined_chars = len(current["text"]) + len(text)
        ends_sentence = current["text"][-1] in _TERMINAL_PUNCT

        # Merge if pause is small, sentence didn't hit a hard stop, and length fits comfortable reading/speaking
        if (
            not ends_sentence
            and gap <= max_gap
            and duration <= max_duration
            and combined_chars <= max_chars
        ):
            sep = "" if current["text"][-1] in "，,、 " else "，"
            current["end"] = round(s["end"], 3)
            current["text"] = f"{current['text']}{sep}{text}"
            current["avg_logprob"] = round(min(current.get("avg_logprob", 0.0), s.get("avg_logprob", 0.0)), 3)
        else:
            merged.append(current)
            current = dict(s)

    if current:
        merged.append(current)

    # Final pass: guarantee strict non-overlap after merging
    for i in range(1, len(merged)):
        if merged[i]["start"] < merged[i - 1]["end"]:
            merged[i]["start"] = round(merged[i - 1]["end"] + 0.02, 3)
            if merged[i]["end"] <= merged[i]["start"]:
                merged[i]["end"] = round(merged[i]["start"] + 0.4, 3)

    return merged


def transcribe_audio_mlx(
    audio_path: str,
    orig_audio_path: Optional[str] = None,
    model_repo: str = DEFAULT_PRIMARY_MODEL,
    merge: bool = True,
    max_gap: float = 0.65,
    max_duration: float = 6.5,
    initial_prompt: str = "以下是普通话节目对话，请使用标准简体中文标点符号。",
) -> List[dict]:
    """
    Transcribes audio on Apple Silicon GPU using MLX Whisper (`whisper-large-v3-turbo` by default)
    with anti-hallucination settings, vocal energy gating, and smart sentence merging.
    If `orig_audio_path` is provided, scans for unvoiced gaps > 3.5s and recovers any speech
    that Demucs vocal separation may have suppressed.
    """
    audio_16k = _load_audio_16k(audio_path)

    result = mlx_whisper.transcribe(
        audio_16k,
        path_or_hf_repo=model_repo,
        language="zh",
        condition_on_previous_text=False,
        word_timestamps=True,
        hallucination_silence_threshold=1.0,
        compression_ratio_threshold=2.0,
        no_speech_threshold=0.55,
        initial_prompt=initial_prompt,
    )
    raw_segments = result.get("segments", [])
    cleaned = _clean_and_gate_segments(raw_segments, audio_16k)
    merged = merge_segments(cleaned, max_gap=max_gap, max_duration=max_duration) if merge else cleaned

    # Gap Recovery Pass (Dual-Stem)
    if orig_audio_path and os.path.exists(orig_audio_path):
        print(f"🔍 Checking for unvoiced gaps using original audio stem: {orig_audio_path}...")
        orig_16k = _load_audio_16k(orig_audio_path)
        gaps_to_check = []

        # Check gap before first segment
        if merged and merged[0]["start"] > 4.0:
            gaps_to_check.append((0.0, merged[0]["start"]))

        # Check gaps between consecutive segments
        for i in range(len(merged) - 1):
            g_start = merged[i]["end"]
            g_end = merged[i+1]["start"]
            if g_end - g_start >= 3.5:
                gaps_to_check.append((g_start, g_end))

        # Check gap after last segment
        total_dur = len(orig_16k) / 16000.0
        if merged and total_dur - merged[-1]["end"] > 4.0:
            gaps_to_check.append((merged[-1]["end"], total_dur))

        recovered_segments = []
        for g_start, g_end in gaps_to_check:
            s_samp = max(0, int(g_start * 16000))
            e_samp = min(len(orig_16k), int(g_end * 16000))
            gap_slice = orig_16k[s_samp:e_samp]
            if len(gap_slice) < 16000:  # < 1s
                continue
            gap_rms = float(np.sqrt(np.mean(gap_slice ** 2)))
            if gap_rms >= 0.012:
                # Vocal energy present in gap! Transcribe this gap slice from original audio
                try:
                    gap_res = mlx_whisper.transcribe(
                        gap_slice,
                        path_or_hf_repo=model_repo,
                        language="zh",
                        condition_on_previous_text=False,
                        initial_prompt=initial_prompt,
                    )
                    sub_raw = gap_res.get("segments", [])
                    for sr_item in sub_raw:
                        sr_item["start"] = round(g_start + sr_item["start"], 3)
                        sr_item["end"] = round(g_start + sr_item["end"], 3)
                    sub_clean = _clean_and_gate_segments(sub_raw, orig_16k, min_rms=0.010)
                    for sc in sub_clean:
                        recovered_segments.append(sc)
                        print(f"  ✨ Recovered dropped speech [{sc['start']:.2f}s -> {sc['end']:.2f}s]: {sc['text']}")
                except Exception as ex:
                    print(f"  ⚠️ Error checking gap {g_start:.1f}-{g_end:.1f}: {ex}")

        if recovered_segments:
            print(f"🎉 Successfully recovered {len(recovered_segments)} missing segments from original audio!")
            all_segs = merged + recovered_segments
            all_segs.sort(key=lambda x: x["start"])
            if merge:
                merged = merge_segments(all_segs, max_gap=max_gap, max_duration=max_duration)
            else:
                merged = all_segs

    return merged


def resubtitle_flagged_segments(
    audio_path: str,
    segments: List[dict],
    flagged_Map: Dict[int, dict],
    brief: Optional[dict] = None,
    escalation_model: str = DEFAULT_ESCALATION_MODEL,
) -> List[int]:
    """
    Re-subtitles specific segment indices that DeepSeek flagged as garbled, homophone-corrupted,
    or suspicious during translation.

    Uses the heavyweight `whisper-large-v3-mlx` model on the padded audio window [start-0.4s, end+0.4s]
    and injects DeepSeek's character glossary + scene hint into `initial_prompt` so Whisper resolves
    domain terms and proper nouns accurately.

    Returns the list of segment indices whose Chinese source text was updated or dropped.
    """
    if not flagged_Map:
        return []

    audio_16k = _load_audio_16k(audio_path)

    # Build a domain glossary prompt from DeepSeek's brief
    glossary_parts = []
    if brief:
        names = [p.get("name", "") for p in (brief.get("people") or []) if p.get("name")]
        terms = [t.get("source", "") for t in (brief.get("terms") or []) if t.get("source")]
        if names:
            glossary_parts.append("人物：" + "、".join(names))
        if terms:
            glossary_parts.append("词汇：" + "、".join(terms))
    base_glossary = "；".join(glossary_parts)

    updated_indices: List[int] = []
    print(f"\n🔍 Re-subtitling {len(flagged_Map)} DeepSeek-flagged segments with [{escalation_model}]...")

    for idx, flag_info in sorted(flagged_Map.items()):
        if idx < 0 or idx >= len(segments):
            continue
        seg = segments[idx]
        old_text = seg["text"]
        reason = flag_info.get("reason", "unclear")
        hint = flag_info.get("hint", "")

        # Pad window slightly to avoid clipped syllables at boundaries
        pad_start = max(0.0, seg["start"] - 0.35)
        pad_end = min(len(audio_16k) / 16000.0, seg["end"] + 0.35)
        if idx > 0:
            pad_start = max(pad_start, segments[idx - 1]["end"] - 0.1)
        if idx + 1 < len(segments):
            pad_end = min(pad_end, segments[idx + 1]["start"] + 0.1)

        s_sample = int(pad_start * 16000)
        e_sample = int(pad_end * 16000)
        clip = audio_16k[s_sample:e_sample]
        if len(clip) < 1600:
            continue

        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms < 0.006:
            print(f"  -> Line #{idx + 1} [{seg['start']:.1f}s]: Pure silence/BGM ({rms=:.4f}), marking empty.")
            seg["text"] = ""
            seg["dropped_hallucination"] = True
            updated_indices.append(idx)
            continue

        # Build context prompt from neighbors + DeepSeek hint + glossary
        prev_zh = segments[idx - 1]["text"] if idx > 0 else ""
        prompt_pieces = [p for p in [base_glossary, hint, prev_zh] if p]
        context_prompt = "。".join(prompt_pieces)[:220] or "请准确转写普通话对话。"

        res = mlx_whisper.transcribe(
            clip,
            path_or_hf_repo=escalation_model,
            language="zh",
            condition_on_previous_text=False,
            temperature=(0.0, 0.2),
            initial_prompt=context_prompt,
        )
        new_text = "，".join(
            s.get("text", "").strip() for s in res.get("segments", []) if s.get("text", "").strip()
        ).strip()

        if not new_text or _is_boilerplate_hallucination(new_text, seg["end"] - seg["start"]):
            print(f"  🧹 Line #{idx + 1} [{seg['start']:.1f}s-{seg['end']:.1f}s]: Confirmed boilerplate/BGM hallucination ('{old_text}'), dropping.")
            seg["text"] = ""
            seg["dropped_hallucination"] = True
            updated_indices.append(idx)
        elif new_text != old_text:
            print(f"  ✨ Line #{idx + 1} [{seg['start']:.1f}s-{seg['end']:.1f}s] ({reason}):")
            print(f"     Before: {old_text}")
            print(f"     After : {new_text}")
            seg["text"] = new_text
            seg["resubtitled"] = True
            updated_indices.append(idx)
        else:
            print(f"  ✓ Line #{idx + 1} [{seg['start']:.1f}s] confirmed by Large-v3: {old_text}")

    return updated_indices


def transcribe_broadcast_stt(
    audio_path: str,
    orig_audio_path: Optional[str] = None,
    engine: str = "qwen3",
    enable_alignment: bool = True,
    enable_name_fixer: bool = True,
    turns_path: Optional[str] = None,
    dossier_path: Optional[str] = None,
    roster_path: Optional[str] = None,
) -> List[dict]:
    """
    Production Broadcast-Grade STT (V2).
    Integrates Diarization-aware turn chunking, Qwen3-ASR/SenseVoice,
    Phonetic Pinyin entity correction, and character-level Forced Alignment with acoustic snapping.
    """
    a_dir = Path(audio_path).parent
    stem = Path(audio_path).stem.replace("_extracted", "")

    # Auto-resolve video-isolated paths if not explicitly given
    resolved_dossier = dossier_path or str(a_dir / f"{stem}_dossier.json")
    resolved_turns = turns_path or str(a_dir / "diar_v2" / "pyannote_turns.json")
    resolved_roster = roster_path or str(a_dir / "face_roster" / "confirmed_actor_roster.json")

    try:
        from src.stt.pipeline import BroadcastSTTPipeline
    except ImportError:
        from CleanProject.src.stt.pipeline import BroadcastSTTPipeline

    pipeline = BroadcastSTTPipeline(
        primary_engine=engine,
        enable_forced_alignment=enable_alignment,
        enable_name_fixer=enable_name_fixer,
        dossier_path=resolved_dossier,
        roster_path=resolved_roster,
        turns_path=resolved_turns,
    )
    res = pipeline.process(
        audio_path=audio_path,
        orig_audio_path=orig_audio_path,
    )
    return res["subtitles"]
