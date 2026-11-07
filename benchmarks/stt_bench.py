"""
stt_bench.py - Broadcast STT Benchmark & Verification Rig for Chinese Reality TV.

Evaluates and compares:
1. Whisper Large-v3 Turbo (Existing Baseline)
2. Qwen3-ASR 0.6B 8-bit
3. Qwen3-ASR 1.7B 8-bit
4. Fun-ASR-Nano 2512
5. SenseVoice-Small
6. FireRedASR2-AED (if downloaded)
7. Full V2 Pipeline (Diarization chunking + Qwen3-ASR + Pinyin fixer + Qwen3 Forced Aligner)

On two representative 4-minute windows:
- W1: 960s - 1200s (High overlap, 12 speakers)
- W2: 420s - 660s (Dense conversational dialogue, 126 cues)

Reports: CER, Filler-Normalized CER, Name Recall, Timestamp Precision, RTF, and Speed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import jiwer
import numpy as np
import opencc
import soundfile as sf

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from CleanProject.src.stt.context import STTContextManager
from CleanProject.src.stt.engines import get_engine
from CleanProject.src.stt.name_fixer import PhoneticNameFixer
from CleanProject.src.stt.pipeline import BroadcastSTTPipeline

t2s = opencc.OpenCC("t2s")
_FILLERS_REGEX = re.compile(r"[啊呢吧嗯哦哈呀对那这个那个啦呗嘛欸]+")
_PUNCT_REGEX = re.compile(r"[^\w\u4e00-\u9fa5]")


def parse_srt_window(srt_path: str, start_time: float, end_time: float) -> List[Dict[str, Any]]:
    """Parses SRT entries falling within [start_time, end_time]."""
    text = Path(srt_path).read_text(encoding="utf-8")
    pattern = re.compile(
        r"(\d+)\s*\n(\d\d):(\d\d):(\d\d),(\d+)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d+)\s*\n(.+?)(?:\n\n|\Z)",
        re.S,
    )
    entries = []
    for m in pattern.finditer(text):
        s_sec = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) + int(m.group(5)) / 1000.0
        e_sec = int(m.group(6)) * 3600 + int(m.group(7)) * 60 + int(m.group(8)) + int(m.group(9)) / 1000.0
        txt = m.group(10).strip()
        if s_sec >= end_time or e_sec <= start_time:
            continue
        entries.append({
            "index": int(m.group(1)),
            "start": round(s_sec, 3),
            "end": round(e_sec, 3),
            "text": txt,
        })
    return entries


def clean_text_for_cer(text: str, strip_fillers: bool = False) -> str:
    """Normalizes Chinese text for CER calculation."""
    s = t2s.convert(text)
    s = _PUNCT_REGEX.sub("", s)
    if strip_fillers:
        s = _FILLERS_REGEX.sub("", s)
    return s.strip().lower()


def compute_name_recall(
    ref_text: str,
    hyp_text: str,
    target_names: List[str],
) -> Tuple[int, int, float]:
    """Computes proper name hits in hypothesis vs reference."""
    ref_clean = t2s.convert(ref_text)
    hyp_clean = t2s.convert(hyp_text)

    total_ref = 0
    total_hyp = 0
    for name in target_names:
        n_s = t2s.convert(name)
        r_c = ref_clean.count(n_s)
        h_c = hyp_clean.count(n_s)
        total_ref += r_c
        total_hyp += min(r_c, h_c)

    recall = (total_hyp / total_ref) if total_ref > 0 else 1.0
    return total_hyp, total_ref, round(recall * 100, 1)


def evaluate_run(
    ref_entries: List[Dict[str, Any]],
    hyp_entries: List[Dict[str, Any]],
    target_names: List[str],
    wall_time: float,
    audio_duration: float,
    engine_label: str,
) -> Dict[str, Any]:
    """Computes full evaluation metrics between reference and hypothesis subtitles."""
    ref_full = "".join(clean_text_for_cer(e["text"], strip_fillers=False) for e in ref_entries)
    hyp_full = "".join(clean_text_for_cer(e["text"], strip_fillers=False) for e in hyp_entries)

    ref_nofill = "".join(clean_text_for_cer(e["text"], strip_fillers=True) for e in ref_entries)
    hyp_nofill = "".join(clean_text_for_cer(e["text"], strip_fillers=True) for e in hyp_entries)

    raw_cer = round(jiwer.cer(ref_full, hyp_full) * 100, 2) if ref_full else 0.0
    norm_cer = round(jiwer.cer(ref_nofill, hyp_nofill) * 100, 2) if ref_nofill else 0.0

    hits, total_names, name_rec = compute_name_recall(ref_full, hyp_full, target_names)

    # Timestamp error analysis (closest cue matching)
    delta_starts = []
    delta_ends = []
    for h in hyp_entries:
        h_s = h["start"]
        h_e = h["end"]
        # Find closest reference cue by start time
        best_diff = 999.0
        best_r = None
        for r in ref_entries:
            diff = abs(r["start"] - h_s)
            if diff < best_diff:
                best_diff = diff
                best_r = r
        if best_r and best_diff <= 3.0:
            delta_starts.append(best_diff)
            delta_ends.append(abs(best_r["end"] - h_e))

    med_ds = round(float(np.median(delta_starts)), 3) if delta_starts else 0.0
    p90_ds = round(float(np.percentile(delta_starts, 90)), 3) if delta_starts else 0.0
    within_150ms = round(sum(1 for d in delta_starts if d <= 0.150) / max(len(delta_starts), 1) * 100, 1)

    rtf = round(wall_time / max(audio_duration, 0.1), 3)

    return {
        "engine": engine_label,
        "audio_dur_s": round(audio_duration, 1),
        "wall_time_s": round(wall_time, 2),
        "rtf": rtf,
        "raw_cer": raw_cer,
        "norm_cer": norm_cer,
        "name_recall": name_rec,
        "name_hits": f"{hits}/{total_names}",
        "med_delta_start_s": med_ds,
        "p90_delta_start_s": p90_ds,
        "sync_within_150ms": within_150ms,
        "hyp_cues": len(hyp_entries),
    }


def run_benchmark(
    vocals_wav: str = "output/ep1_lower_30m_stems/htdemucs/ep1_lower_30m_24k/vocals.wav",
    ref_srt: str = "output/ep1_lower_30m_chinese.srt",
    output_dir: str = "output/stt_bench",
    engines: Optional[List[str]] = None,
    windows: Optional[List[Tuple[float, float, str]]] = None,
) -> None:
    """Executes the STT benchmark matrix and saves markdown report + json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx_mgr = STTContextManager()
    target_names = ctx_mgr.get_all_names()

    # Default evaluation windows:
    # W1: 960s - 1200s (240s, 12 speakers, heavy overlap)
    # W2: 420s - 660s (240s, dense dialogue, 126 cues)
    test_windows = windows or [
        (960.0, 1200.0, "W1_Overlap_960-1200s"),
        (420.0, 660.0, "W2_Dense_420-660s"),
    ]

    candidate_engines = engines or [
        "whisper-turbo",
        "sensevoice",
        "funasr-nano",
        "qwen3-0.6b",
        "qwen3-1.7b",
        "firered",
        "v2-pipeline",
    ]

    all_results = []
    print("=" * 70)
    print("🎙️ STARTING BROADCAST CHINESE STT BENCHMARK ON APPLE SILICON")
    print(f"📁 Vocals audio: {vocals_wav}")
    print(f"📜 Ground truth: {ref_srt}")
    print(f"🧪 Engines     : {candidate_engines}")
    print("=" * 70)

    for w_start, w_end, w_name in test_windows:
        w_dur = w_end - w_start
        print(f"\n==================== WINDOW: {w_name} ({w_dur:.1f}s) ====================")
        ref_entries = parse_srt_window(ref_srt, w_start, w_end)
        print(f"📖 Reference broadcast cues: {len(ref_entries)} cues")

        # Load audio slice for this window
        full_audio, sr = sf.read(vocals_wav, dtype="float32")
        if full_audio.ndim > 1:
            full_audio = np.mean(full_audio, axis=1)
        if sr != 16000:
            import librosa
            full_audio = librosa.resample(full_audio, orig_sr=sr, target_sr=16000)

        s_samp = int(w_start * 16000)
        e_samp = min(len(full_audio), int(w_end * 16000))
        win_audio = full_audio[s_samp:e_samp]

        for eng_name in candidate_engines:
            print(f"\n--- Testing Engine: [{eng_name}] on {w_name} ---")
            t_start = time.time()
            hyp_entries = []

            try:
                if eng_name == "v2-pipeline":
                    # Full production V2 pipeline: chunker + Qwen3-ASR + Pinyin Fixer + Forced Aligner
                    pipeline = BroadcastSTTPipeline(
                        primary_engine="qwen3",
                        enable_forced_alignment=True,
                        enable_name_fixer=True,
                    )
                    res = pipeline.process(
                        audio_path=vocals_wav,
                        start_time=w_start,
                        end_time=w_end,
                    )
                    hyp_entries = res["subtitles"]
                    # Save SRT for inspection
                    srt_out = out_dir / f"{w_name}_{eng_name}.srt"
                    pipeline.export_srt(hyp_entries, str(srt_out))

                elif eng_name == "whisper-turbo":
                    # Baseline MLX Whisper
                    import mlx_whisper
                    res = mlx_whisper.transcribe(
                        win_audio,
                        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                        language="zh",
                        condition_on_previous_text=False,
                        initial_prompt="以下是普通话对话，请使用标准简体中文标点符号。",
                    )
                    for s in res.get("segments", []):
                        hyp_entries.append({
                            "start": round(w_start + float(s.get("start", 0.0)), 3),
                            "end": round(w_start + float(s.get("end", 0.0)), 3),
                            "text": (s.get("text") or "").strip(),
                        })

                else:
                    # Direct engine run
                    eng = get_engine(eng_name)
                    res = eng.transcribe(
                        win_audio,
                        hotwords=ctx_mgr.get_hotwords(),
                        system_prompt=ctx_mgr.get_system_prompt(),
                    )
                    eng.unload()
                    for s in res:
                        hyp_entries.append({
                            "start": round(w_start + float(s.get("start", 0.0)), 3),
                            "end": round(w_start + float(s.get("end", 0.0)), 3),
                            "text": s.get("text", "").strip(),
                        })

                wall_time = time.time() - t_start
                stats = evaluate_run(
                    ref_entries=ref_entries,
                    hyp_entries=hyp_entries,
                    target_names=target_names,
                    wall_time=wall_time,
                    audio_duration=w_dur,
                    engine_label=eng_name,
                )
                stats["window"] = w_name
                all_results.append(stats)

                print(f"   CER (Raw): {stats['raw_cer']}% | CER (No Fillers): {stats['norm_cer']}%")
                print(f"   Name Recall: {stats['name_recall']}% ({stats['name_hits']}) | RTF: {stats['rtf']}x | Time: {wall_time:.1f}s")
                print(f"   Timestamp Sync <=150ms: {stats['sync_within_150ms']}% (Median start diff: {stats['med_delta_start_s']}s)")

            except Exception as e:
                print(f"❌ Error testing {eng_name} on {w_name}: {e}")
                import traceback
                traceback.print_exc()

    # Save results to JSON and Markdown report
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    report_md = generate_markdown_report(all_results)
    report_path = out_dir / "benchmark_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"📊 Benchmark report saved -> {report_path}")
    print("=" * 70)
    print(report_md)


def generate_markdown_report(results: List[Dict[str, Any]]) -> str:
    """Generates a clean GFM markdown table of benchmark results."""
    lines = [
        "# Broadcast Chinese STT Benchmark Report (Apple Silicon M4)",
        "",
        "| Window | Engine | CER (Raw) | CER (Norm) | Name Recall | Median ΔStart | Sync ≤150ms | RTF | Wall Time |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.get('window')} | **{r.get('engine')}** | {r.get('raw_cer')}% | {r.get('norm_cer')}% | {r.get('name_recall')}% ({r.get('name_hits')}) | {r.get('med_delta_start_s')}s | {r.get('sync_within_150ms')}% | {r.get('rtf')}x | {r.get('wall_time_s')}s |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Broadcast STT Benchmark")
    parser.add_argument("--vocals", default="output/ep1_lower_30m_stems/htdemucs/ep1_lower_30m_24k/vocals.wav")
    parser.add_argument("--ref-srt", default="output/ep1_lower_30m_chinese.srt")
    parser.add_argument("--output-dir", default="output/stt_bench")
    args = parser.parse_args()

    run_benchmark(vocals_wav=args.vocals, ref_srt=args.ref_srt, output_dir=args.output_dir)
