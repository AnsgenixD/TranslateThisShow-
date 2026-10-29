import mlx_whisper
from pathlib import Path

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_audio_mlx(audio_path: str, model_repo: str = "mlx-community/whisper-medium-mlx") -> list:
    """
    Transcribes audio on Apple Silicon GPU using MLX.
    Returns list of dicts: [{'start': ..., 'end': ..., 'text': ...}]
    """
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        language="zh"
    )
    return result["segments"]

def write_srt(segments: list, output_srt: str):
    p = Path(output_srt)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            text = seg.get("translated_text", seg.get("text", "")).strip()
            start_str = format_timestamp(seg["start"])
            end_str = format_timestamp(seg["end"])
            f.write(f"{idx}\n{start_str} --> {end_str}\n{text}\n\n")
