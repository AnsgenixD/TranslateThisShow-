import subprocess
from pathlib import Path

def extract_audio(video_path: str, output_wav: str, sample_rate: int = 16000) -> str:
    """
    Extracts 16-bit uncompressed PCM mono WAV from video.
    """
    out = Path(output_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(out)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)

def merge_video_subtitles(video_path: str, srt_path: str, output_path: str) -> str:
    """
    Embeds English subtitles into the MP4 container as a switchable subtitle track (mov_text).
    This completes in 0.01 seconds without re-encoding video pixels!
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(srt_path),
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        str(out)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)
