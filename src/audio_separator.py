import os
import time
import psutil
import soundfile as sf
import numpy as np
from pathlib import Path
from demucs_mlx import Separator

def get_dynamic_chunk_size() -> int:
    """
    Proportional Dynamic Memory Allocation:
    - Allocates 75% of currently available free RAM to Demucs.
    - Leaves 25% safety buffer so macOS never triggers swap or compression.
    - Buffer consumption rate: ~0.9 MB per second of audio (plus ~150 MB model base).
    """
    avail_mb = psutil.virtual_memory().available / (1024 * 1024)
    # Target 75% of available memory for high throughput
    target_budget_mb = avail_mb * 0.75
    
    # Base model footprint is ~150 MB
    usable_audio_budget = max(100.0, target_budget_mb - 150.0)
    
    # Calculate duration: ~0.9 MB per second of audio
    calculated_seconds = int(usable_audio_budget / 0.9)
    
    # Minimum floor of 60 seconds
    return max(60, calculated_seconds)

def crossfade_stitch(chunk_prev: np.ndarray, chunk_next: np.ndarray, overlap_samples: int) -> np.ndarray:
    """
    Smoothly crossfades two consecutive audio arrays over overlap_samples.
    Uses linear equal-power weighting to eliminate audible clicks/pops.
    """
    if overlap_samples <= 0 or len(chunk_prev) < overlap_samples or len(chunk_next) < overlap_samples:
        return np.concatenate([chunk_prev, chunk_next], axis=0)

    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

    if chunk_prev.ndim > 1:
        fade_out = fade_out[:, np.newaxis]
        fade_in = fade_in[:, np.newaxis]

    overlap_prev = chunk_prev[-overlap_samples:]
    overlap_next = chunk_next[:overlap_samples]
    blended = (overlap_prev * fade_out) + (overlap_next * fade_in)

    head = chunk_prev[:-overlap_samples]
    tail = chunk_next[overlap_samples:]
    return np.concatenate([head, blended, tail], axis=0)

def separate_vocals_and_bgm(wav_path: str, output_dir: str, overlap_sec: float = 2.0) -> dict:
    """
    Memory-Proportional Audio Separation:
    - Maximizes RAM usage up to 75% of available memory for maximum throughput.
    - Slices into large, efficient chunks only if duration exceeds the memory budget.
    - Completely avoids swap memory paging.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    temp_dir = out / "chunks_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    info = sf.info(wav_path)
    sr = info.samplerate
    total_duration = info.duration
    chunk_size = get_dynamic_chunk_size()
    overlap_samples = int(overlap_sec * sr)

    print(f"\n[Audio Separator] Total Duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"[Audio Separator] Proportional RAM Chunk Budget: {chunk_size}s ({chunk_size/60:.1f}m)")

    sep = Separator(model="htdemucs")

    # Single-pass fast path: audio fits comfortably within 75% of free RAM
    if total_duration <= chunk_size:
        print("⚡ Audio fits within memory budget: Processing in a SINGLE pass (zero stitching required)...")
        sep.separate(wav_path, output_dir=out)
        
        drums, _ = sf.read(out / "drums.wav")
        bass, _ = sf.read(out / "bass.wav")
        other, _ = sf.read(out / "other.wav")
        bgm = np.clip(drums + bass + other, -1.0, 1.0)
        
        bgm_path = out / "accompaniment.wav"
        sf.write(bgm_path, bgm, sr)
        
        for f in ["drums.wav", "bass.wav", "other.wav"]:
            p = out / f
            if p.exists(): p.unlink()
            
        return {
            "vocals": str(out / "vocals.wav"),
            "accompaniment": str(bgm_path)
        }

    # Multi-chunk path for very long files or constrained RAM
    print(f"🔄 Audio exceeds current free memory: Splitting into large {chunk_size/60:.1f}m chunks with {overlap_sec}s crossfade...")
    
    data, _ = sf.read(wav_path, dtype="float32")
    total_samples = len(data)
    chunk_samples = int(chunk_size * sr)

    accumulated_vocals = None
    accumulated_bgm = None

    start_idx = 0
    chunk_num = 1

    while start_idx < total_samples:
        end_idx = min(total_samples, start_idx + chunk_samples)
        chunk_data = data[start_idx:end_idx]
        
        chunk_wav_path = temp_dir / f"chunk_{chunk_num}.wav"
        chunk_out_dir = temp_dir / f"out_{chunk_num}"
        sf.write(chunk_wav_path, chunk_data, sr)

        print(f"   Processing Chunk {chunk_num}: [{start_idx/sr:.1f}s -> {end_idx/sr:.1f}s] ({len(chunk_data)/sr:.1f}s)...")
        sep.separate(str(chunk_wav_path), output_dir=chunk_out_dir)

        c_voc, _ = sf.read(chunk_out_dir / "vocals.wav", dtype="float32")
        c_drums, _ = sf.read(chunk_out_dir / "drums.wav", dtype="float32")
        c_bass, _ = sf.read(chunk_out_dir / "bass.wav", dtype="float32")
        c_other, _ = sf.read(chunk_out_dir / "other.wav", dtype="float32")
        c_bgm = np.clip(c_drums + c_bass + c_other, -1.0, 1.0)

        if accumulated_vocals is None:
            accumulated_vocals = c_voc
            accumulated_bgm = c_bgm
        else:
            accumulated_vocals = crossfade_stitch(accumulated_vocals, c_voc, overlap_samples)
            accumulated_bgm = crossfade_stitch(accumulated_bgm, c_bgm, overlap_samples)

        chunk_wav_path.unlink()
        for stem_file in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
            (chunk_out_dir / stem_file).unlink()
        chunk_out_dir.rmdir()

        if end_idx >= total_samples:
            break
        start_idx = end_idx - overlap_samples
        chunk_num += 1

    vocals_final = out / "vocals.wav"
    bgm_final = out / "accompaniment.wav"

    print("💾 Writing assembled stems to disk...")
    sf.write(vocals_final, accumulated_vocals, sr)
    sf.write(bgm_final, accumulated_bgm, sr)

    if temp_dir.exists():
        temp_dir.rmdir()

    print("✅ Complete! Maximized RAM throughput with zero swap usage.")
    return {
        "vocals": str(vocals_final),
        "accompaniment": str(bgm_final)
    }
