"""
dubber.py - Fast Multi-Speaker Neural Video Dubbing Engine.

Features:
1. Multi-voice neural speech synthesis powered by Microsoft Edge-TTS (14+ calibrated voices).
2. Personality/gender-matched voice assignment for detected actors.
3. Dynamic sentence-tempo scaling and acoustic headroom matching (zero word truncation).
4. Studio mixing: Demucs BGM preservation with steady, natural background levels.
5. Lossless video stream muxing (`-c:v copy -c:a aac`) via FFmpeg.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf

DEFAULT_FEMALE_VOICES = [
    "en-US-JennyNeural",
    "en-US-AvaNeural",
    "en-US-EmmaNeural",
    "en-GB-SoniaNeural",
    "en-US-MichelleNeural",
    "en-US-AnaNeural",
    "en-GB-MaisieNeural",
    "en-US-AriaNeural",
]
DEFAULT_MALE_VOICES = [
    "en-US-GuyNeural",
    "en-US-AndrewNeural",
    "en-US-ChristopherNeural",
    "en-US-BrianNeural",
    "en-US-EricNeural",
    "en-GB-RyanNeural",
]


def get_or_create_actor_voices(
    actor_ids: List[str],
    voice_file: Optional[Path] = None,
    verified_genders: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Dynamically assigns personality-matched neural voices based on verified actor gender."""
    existing_map: Dict[str, str] = {}
    if voice_file and voice_file.exists():
        try:
            with open(voice_file, "r", encoding="utf-8") as f:
                existing_map = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load existing voice file: {e}")

    verified = verified_genders or {}
    voice_map = dict(existing_map)
    f_idx = 0
    m_idx = 0

    for aid in sorted(actor_ids):
        if aid in voice_map:
            continue

        aid_upper = aid.upper()
        if "MALE" in aid_upper or aid_upper == "NARRATOR":
            gender = "male"
        elif "FEMALE" in aid_upper:
            gender = "female"
        else:
            gender = verified.get(aid, "female" if (f_idx <= m_idx) else "male")

        if gender == "female":
            voice = DEFAULT_FEMALE_VOICES[f_idx % len(DEFAULT_FEMALE_VOICES)]
            f_idx += 1
        else:
            voice = DEFAULT_MALE_VOICES[m_idx % len(DEFAULT_MALE_VOICES)]
            m_idx += 1
        voice_map[aid] = voice

    if voice_file:
        voice_file.parent.mkdir(parents=True, exist_ok=True)
        with open(voice_file, "w", encoding="utf-8") as f:
            json.dump(voice_map, f, indent=2)

    return voice_map


def strip_silence(audio: np.ndarray, sr: int, threshold: float = 0.008, pad_sec: float = 0.08) -> np.ndarray:
    """Removes dead leading/trailing silence while preserving delicate final consonants."""
    if len(audio) == 0:
        return audio
    mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    non_silent = np.where(np.abs(mono) > threshold)[0]
    if len(non_silent) == 0:
        return audio

    pad_samples = int(pad_sec * sr)
    start = max(0, non_silent[0] - pad_samples)
    end = min(len(audio), non_silent[-1] + pad_samples)
    trimmed = audio[start:end]

    fade_len = min(int(0.010 * sr), len(trimmed) // 4)
    if fade_len > 0:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
        if trimmed.ndim > 1:
            trimmed[:fade_len] *= fade_in[:, np.newaxis]
            trimmed[-fade_len:] *= fade_out[:, np.newaxis]
        else:
            trimmed[:fade_len] *= fade_in
            trimmed[-fade_len:] *= fade_out

    return trimmed


async def synthesize_line(
    sem: asyncio.Semaphore,
    seg: Dict[str, Any],
    out_dir: Path,
    voice_map: Dict[str, str],
    target_sr: int = 24000,
    force_resynth: bool = False,
    max_retries: int = 3,
) -> Tuple[int, str]:
    import edge_tts
    import librosa

    idx = seg.get("index", 0)
    text = (seg.get("en") or seg.get("translated_text") or "").strip()
    if not text:
        text = (seg.get("zh") or seg.get("text") or "").strip()
    if not text or not re.search(r"[\w\u4e00-\u9fff]", text):
        return idx, ""

    actor_id = seg.get("actor_id") or seg.get("speaker") or seg.get("cluster_id") or "Speaker_00"
    voice = voice_map.get(actor_id, DEFAULT_FEMALE_VOICES[0])

    slot_dur = max(0.2, float(seg["end"]) - float(seg["start"]))
    headroom_dur = seg.get("headroom_dur", slot_dur)

    word_count = max(1, len(text.split()))
    est_dur = (word_count / 2.5) + 0.15

    rate_str = "+0%"
    if est_dur > headroom_dur and headroom_dur >= 0.3:
        needed_speedup = est_dur / headroom_dur
        rate_pct = int(min(45, max(5, (needed_speedup - 1.0) * 100)))
        rate_str = f"+{rate_pct}%"
    elif est_dur > slot_dur * 1.15 and slot_dur >= 0.3:
        mild_speedup = est_dur / slot_dur
        rate_pct = int(min(25, max(5, (mild_speedup - 1.0) * 50)))
        rate_str = f"+{rate_pct}%"

    mp3_path = out_dir / f"line_{idx:04d}.mp3"
    wav_path = out_dir / f"line_{idx:04d}.wav"

    if not force_resynth and wav_path.exists() and wav_path.stat().st_size > 1000:
        return idx, str(wav_path)

    async with sem:
        for attempt in range(max_retries):
            try:
                comm = edge_tts.Communicate(text, voice, rate=rate_str)
                await comm.save(str(mp3_path))

                if not mp3_path.exists() or mp3_path.stat().st_size < 100:
                    raise IOError(f"Empty mp3 output for line {idx}")

                data, sr = sf.read(str(mp3_path), dtype="float32")
                if mp3_path.exists():
                    mp3_path.unlink()

                trimmed = strip_silence(data, sr, threshold=0.008, pad_sec=0.08)
                if len(trimmed) == 0:
                    trimmed = data

                if sr != target_sr:
                    trimmed = librosa.resample(trimmed, orig_sr=sr, target_sr=target_sr)

                # Enforce strict slot headroom so one speaker never spills into the next speaker's turn
                actual_dur = len(trimmed) / float(target_sr)
                if actual_dur > headroom_dur and headroom_dur >= 0.35:
                    stretch_rate = min(1.45, actual_dur / (headroom_dur * 0.96))
                    if stretch_rate > 1.03:
                        trimmed = librosa.effects.time_stretch(trimmed, rate=stretch_rate)

                sf.write(str(wav_path), trimmed, target_sr)
                return idx, str(wav_path)
            except Exception as e:
                if mp3_path.exists():
                    mp3_path.unlink()
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    print(f"  ❌ Line {idx} synthesis failed: {e}")
                    return idx, ""
    return idx, ""


async def run_batch_synthesis(
    segments: List[Dict[str, Any]],
    out_dir: Path,
    voice_map: Dict[str, str],
    concurrency: int = 5,
    force_resynth: bool = False,
) -> Dict[int, str]:
    for i, s in enumerate(segments):
        slot_dur = float(s["end"]) - float(s["start"])
        if i + 1 < len(segments):
            nxt_start = float(segments[i + 1]["start"])
            headroom = max(slot_dur, nxt_start - float(s["start"]) - 0.10)
        else:
            headroom = slot_dur + 5.0
        s["headroom_dur"] = headroom

    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    t0 = time.time()
    tasks = [
        synthesize_line(
            sem, s, out_dir, voice_map=voice_map, force_resynth=force_resynth, max_retries=3
        )
        for s in segments
    ]
    results: Dict[int, str] = {}
    completed = 0

    for fut in asyncio.as_completed(tasks):
        idx, wav_p = await fut
        results[idx] = wav_p
        completed += 1
        if completed % 50 == 0 or completed == len(segments):
            elapsed = time.time() - t0
            print(f"  -> Generated {completed}/{len(segments)} lines ({elapsed:.1f}s)")

    return results


def studio_mix_and_render(
    segments: List[Dict[str, Any]],
    wav_paths: Dict[int, str],
    bgm_path: Optional[str],
    output_wav: str,
    target_sr: int = 24000,
    total_duration_sec: Optional[float] = None,
    bgm_volume: float = 0.80,
) -> None:
    """Combines synthesized speech lines onto a timeline and mixes with Demucs BGM stem."""
    if total_duration_sec is None:
        max_end = max((float(s["end"]) for s in segments), default=60.0)
        total_duration_sec = max_end + 5.0

    total_samples = int(total_duration_sec * target_sr)
    speech_track = np.zeros(total_samples, dtype=np.float32)
    global_prev_end: float = 0.0
    active_count = 0

    for seg in segments:
        idx = seg.get("index", 0)
        wav_p = wav_paths.get(idx)
        if not wav_p or not os.path.exists(wav_p):
            continue

        clip, sr = sf.read(wav_p, dtype="float32")
        if clip.ndim > 1:
            clip = np.mean(clip, axis=1)

        start_sec = float(seg["start"])
        if start_sec < global_prev_end:
            start_sec = global_prev_end + 0.04

        s_idx = max(0, int(start_sec * target_sr))
        e_idx = min(total_samples, s_idx + len(clip))
        clip_len = e_idx - s_idx

        if clip_len > 0:
            speech_track[s_idx:e_idx] += clip[:clip_len]
            actual_end_sec = start_sec + (clip_len / target_sr)
            global_prev_end = max(global_prev_end, actual_end_sec)
            active_count += 1

    print(f"🎙️ Placed {active_count}/{len(segments)} dialogue clips onto speech master track.")

    # Mix BGM
    if bgm_path and os.path.exists(bgm_path):
        bgm_full, bgm_sr = sf.read(bgm_path, dtype="float32")
        if bgm_full.ndim > 1:
            bgm_full = np.mean(bgm_full, axis=1)

        if len(bgm_full) < total_samples:
            bgm_pad = np.zeros(total_samples, dtype=np.float32)
            bgm_pad[:len(bgm_full)] = bgm_full
            bgm_full = bgm_pad
        else:
            bgm_full = bgm_full[:total_samples]
        bgm_stem = bgm_full * bgm_volume
    else:
        bgm_stem = np.zeros(total_samples, dtype=np.float32)

    master = speech_track + bgm_stem
    peak = float(np.max(np.abs(master)))
    if peak > 0.95:
        master = master * (0.95 / peak)

    sf.write(output_wav, master, target_sr)
    print(f"💾 Dubbed master audio saved: {output_wav}")


def mux_video(video_in: str, audio_in: str, video_out: str, srt_in: Optional[str] = None) -> None:
    """Lossless video multiplexing (`-c:v copy -c:a aac`) with optional switchable English subtitles."""
    if srt_in and os.path.exists(srt_in):
        cmd = [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", audio_in,
            "-i", srt_in,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-map", "2:s:0",
            "-shortest",
            video_out
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_in,
            "-i", audio_in,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            video_out
        ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Final Dubbed Video: {video_out}")


def merge_utterances_for_dubbing(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges consecutive subtitle cues belonging to the same speaker into coherent
    sentence-level dubbing utterances when separated by a short breath pause (<= 0.65s)
    and non-terminal punctuation (or short clause duration), preventing mid-sentence
    prosody/tempo resets in TTS synthesis.
    """
    if not segments:
        return []

    merged: List[Dict[str, Any]] = []
    cur = dict(segments[0])
    cur["en"] = (cur.get("en") or cur.get("translated_text") or "").strip()

    for nxt in segments[1:]:
        nxt_en = (nxt.get("en") or nxt.get("translated_text") or "").strip()
        cur_spk = cur.get("actor_id") or cur.get("speaker") or cur.get("cluster_id") or "Speaker_00"
        nxt_spk = nxt.get("actor_id") or nxt.get("speaker") or nxt.get("cluster_id") or "Speaker_00"
        same_spk = (cur_spk == nxt_spk)
        gap = float(nxt["start"]) - float(cur["end"])
        comb_dur = float(nxt["end"]) - float(cur["start"])
        cur_en = cur["en"]

        is_cont = (
            cur_en.endswith((",", "...", "—", "-"))
            or (nxt_en and nxt_en[0].islower())
            or (float(cur["end"]) - float(cur["start"]) < 1.2)
        )

        if same_spk and gap <= 0.65 and comb_dur <= 9.0 and is_cont:
            if cur_en.endswith("...") and nxt_en and nxt_en[0].islower():
                cur_en = cur_en[:-3].rstrip()
            cur["en"] = f"{cur_en} {nxt_en}".strip()
            cur["translated_text"] = cur["en"]
            cur["end"] = nxt["end"]
        else:
            merged.append(cur)
            cur = dict(nxt)
            cur["en"] = nxt_en

    merged.append(cur)
    for i, m in enumerate(merged, 1):
        m["index"] = i
    return merged


def dub_video_pipeline(
    segments: List[Dict[str, Any]],
    video_path: str,
    output_mp4: str,
    bgm_path: Optional[str] = None,
    clips_dir: str = "output/dubbing_clips",
    voice_map_file: Optional[str] = None,
    verified_genders: Optional[Dict[str, str]] = None,
    srt_path: Optional[str] = None,
    force_resynth: bool = False,
) -> bool:
    """
    High-level entry point for neural video dubbing.
    Returns True on success, False if required dependencies are missing.
    """
    try:
        import edge_tts
        import librosa
    except ImportError:
        print("\n💡 [Optional Feature] Neural Dubbing requires 'edge-tts' and 'librosa'.")
        print("   To enable multi-speaker neural English dubbing:")
        print("     pip install edge-tts librosa")
        print("   Supported voices:")
        print("     - 14+ Microsoft Neural Voices (US & UK accents, gender-calibrated)")
        print("   Skipping neural audio dubbing for now...\n")
        return False

    out_clips = Path(clips_dir)
    if force_resynth and out_clips.exists():
        for old_wav in out_clips.glob("line_*.wav"):
            try:
                old_wav.unlink()
            except Exception:
                pass

    dub_segments = merge_utterances_for_dubbing(segments)
    actor_ids = sorted(
        list(set(s.get("actor_id") or s.get("speaker") or s.get("cluster_id") or "Speaker_00" for s in dub_segments))
    )
    v_file = Path(voice_map_file) if voice_map_file else None
    voice_map = get_or_create_actor_voices(actor_ids, voice_file=v_file, verified_genders=verified_genders)

    print("🎭 Assigned Speaker Voices:")
    for aid, vname in voice_map.items():
        print(f"   • {aid} -> {vname}")

    print(f"\n⚡ Synthesizing {len(dub_segments)} coherent utterances (from {len(segments)} cues) across {len(voice_map)} speaker voices...")
    wav_paths = asyncio.run(
        run_batch_synthesis(
            segments=dub_segments,
            out_dir=out_clips,
            voice_map=voice_map,
            concurrency=5,
            force_resynth=force_resynth,
        )
    )

    dubbed_wav = str(Path(output_mp4).with_suffix(".wav"))
    studio_mix_and_render(
        segments=dub_segments,
        wav_paths=wav_paths,
        bgm_path=bgm_path,
        output_wav=dubbed_wav,
    )

    mux_video(video_path, dubbed_wav, output_mp4, srt_in=srt_path)
    return True
