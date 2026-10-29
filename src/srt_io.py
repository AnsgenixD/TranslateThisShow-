"""Tiny SRT reader/writer shared by every stage of the pipeline."""
import re
from pathlib import Path

_TIMESTAMP = re.compile(r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})")


def format_timestamp(seconds: float) -> str:
    """76.167 -> '00:01:16,167'. Works in whole milliseconds to avoid rounding glitches."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _to_seconds(match: "re.Match") -> float:
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def read_srt(path: str) -> list:
    """
    Reads an .srt file into a list of {"start": float, "end": float, "text": str}.
    Tolerates a BOM, Windows/Mac line endings, a missing index line, and multi-line subtitles.
    """
    raw = Path(path).read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    segments = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            if "-->" not in line:
                continue
            stamps = list(_TIMESTAMP.finditer(line))
            if len(stamps) >= 2:
                text = " ".join(l.strip() for l in lines[i + 1:] if l.strip())
                segments.append({
                    "start": _to_seconds(stamps[0]),
                    "end": _to_seconds(stamps[1]),
                    "text": text,
                })
            break
    return segments


def write_srt(segments: list, output_srt: str, text_key: str = "text") -> None:
    """Writes segments to an .srt file, using seg[text_key] as the subtitle text."""
    p = Path(output_srt)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            text = str(seg.get(text_key, "")).strip()
            f.write(f"{idx}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{text}\n\n")
