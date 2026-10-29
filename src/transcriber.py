import mlx_whisper


def transcribe_audio_mlx(audio_path: str, model_repo: str = "mlx-community/whisper-medium-mlx") -> list:
    """
    Transcribes audio on the Apple Silicon GPU using MLX.
    Returns a list of dicts: [{'start': ..., 'end': ..., 'text': ...}, ...]
    """
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        language="zh"
    )
    return result["segments"]
