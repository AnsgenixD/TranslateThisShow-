"""
segmenter.py - Semantic Clause Segmentation for Subtitles and Neural Dubbing.

Produces two synchronized outputs from ASR tokens:
1. Broadcast Subtitle Cues: 0.8s - 6.0s, max 18 chars/line, non-overlapping, strictly speaker-isolated.
2. Neural Dubbing Units: Sentence-level units (up to 8.5s) optimized for Edge-TTS tempo fitting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_SENTENCE_TERMINATORS = set("。？！?!…\n")
_CLAUSE_SEPARATORS = set("，,、；; ")


class SemanticSegmenter:
    """
    Transforms raw transcribed or aligned segments into production broadcast subtitles
    and dubbing units.
    """

    def __init__(
        self,
        min_sub_duration: float = 0.8,
        max_sub_duration: float = 6.0,
        max_sub_chars: int = 34,
        max_chars_per_line: int = 18,
        clause_gap_threshold: float = 0.45,
        dub_max_duration: float = 8.5,
        dub_max_gap: float = 0.85,
    ):
        self.min_sub_duration = min_sub_duration
        self.max_sub_duration = max_sub_duration
        self.max_sub_chars = max_sub_chars
        self.max_chars_per_line = max_chars_per_line
        self.clause_gap_threshold = clause_gap_threshold
        self.dub_max_duration = dub_max_duration
        self.dub_max_gap = dub_max_gap

    def format_subtitle_lines(self, text: str) -> str:
        """Splits long text across at most two balanced lines."""
        text = text.strip()
        if len(text) <= self.max_chars_per_line:
            return text

        # Find best split point near the middle (prefer punctuation)
        mid = len(text) // 2
        best_idx = -1
        best_dist = 999

        for p_idx, char in enumerate(text):
            if char in "，,、；; ":
                dist = abs(p_idx - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = p_idx

        if best_idx != -1 and 4 <= best_idx <= len(text) - 4:
            line1 = text[:best_idx + 1].strip()
            line2 = text[best_idx + 1:].strip()
            return f"{line1}\n{line2}"

        # Hard break near middle
        return f"{text[:mid]}\n{text[mid:]}"

    def build_subtitles(self, raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Builds broadcast-grade non-overlapping subtitle cues.
        """
        if not raw_segments:
            return []

        # Filter out empty or micro-duration artifacts
        valid = []
        for s in raw_segments:
            txt = (s.get("text") or "").strip()
            if not txt:
                continue
            st = float(s.get("start", 0.0))
            et = float(s.get("end", st + 0.1))
            if et <= st + 0.05:
                continue
            valid.append({
                "start": round(st, 3),
                "end": round(et, 3),
                "text": txt,
                "cluster_id": s.get("cluster_id") or s.get("speaker") or "SPEAKER_00",
                "events": s.get("events", []),
            })

        if not valid:
            return []

        # Step 1: Merge sub-clauses of the same speaker if gap is small and length fits
        merged: List[Dict[str, Any]] = []
        cur = dict(valid[0])

        for nxt in valid[1:]:
            gap = nxt["start"] - cur["end"]
            dur = nxt["end"] - cur["start"]
            combined_len = len(cur["text"]) + len(nxt["text"])
            same_spk = (cur["cluster_id"] == nxt["cluster_id"])
            ends_term = cur["text"][-1] in _SENTENCE_TERMINATORS

            if (
                same_spk
                and not ends_term
                and 0.0 <= gap <= self.clause_gap_threshold
                and dur <= self.max_sub_duration
                and combined_len <= self.max_sub_chars
            ):
                sep = "" if cur["text"][-1] in _CLAUSE_SEPARATORS else "，"
                cur["text"] = f"{cur['text']}{sep}{nxt['text']}"
                cur["end"] = nxt["end"]
            else:
                merged.append(cur)
                cur = dict(nxt)

        merged.append(cur)

        # Step 2: Enforce minimum duration and non-overlapping monotonicity
        final_subs: List[Dict[str, Any]] = []
        for i, cue in enumerate(merged):
            st = cue["start"]
            et = cue["end"]

            # Expand tiny durations up to min_sub_duration if there is room
            if et - st < self.min_sub_duration:
                et = round(st + self.min_sub_duration, 3)

            # Enforce non-overlap with previous cue
            if final_subs:
                prev_et = final_subs[-1]["end"]
                if st < prev_et:
                    # Clip or nudge
                    if et <= prev_et + 0.1:
                        # Subsumed, skip
                        continue
                    st = round(prev_et + 0.04, 3)
                    if et <= st:
                        et = round(st + self.min_sub_duration, 3)

            formatted_text = self.format_subtitle_lines(cue["text"])
            final_subs.append({
                "index": len(final_subs) + 1,
                "start": round(st, 3),
                "end": round(et, 3),
                "text": formatted_text,
                "raw_text": cue["text"],
                "cluster_id": cue["cluster_id"],
                "events": cue.get("events", []),
            })

        return final_subs

    def build_dubbing_units(self, raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Builds sentence-complete dubbing units for Edge-TTS neural dubbing.
        """
        if not raw_segments:
            return []

        units: List[Dict[str, Any]] = []
        cur: Optional[Dict[str, Any]] = None

        for s in raw_segments:
            txt = (s.get("text") or "").strip()
            if not txt:
                continue

            st = float(s.get("start", 0.0))
            et = float(s.get("end", st + 0.1))
            spk = s.get("cluster_id") or s.get("speaker") or "SPEAKER_00"

            if cur is None:
                cur = {
                    "start": st,
                    "end": et,
                    "text": txt,
                    "cluster_id": spk,
                    "sub_segments": [s],
                }
                continue

            gap = st - cur["end"]
            dur = et - cur["start"]
            same_spk = (cur["cluster_id"] == spk)
            ends_term = cur["text"][-1] in _SENTENCE_TERMINATORS

            if (
                same_spk
                and not ends_term
                and gap <= self.dub_max_gap
                and dur <= self.dub_max_duration
            ):
                sep = "" if cur["text"][-1] in _CLAUSE_SEPARATORS else "，"
                cur["text"] = f"{cur['text']}{sep}{txt}"
                cur["end"] = et
                cur["sub_segments"].append(s)
            else:
                units.append(cur)
                cur = {
                    "start": st,
                    "end": et,
                    "text": txt,
                    "cluster_id": spk,
                    "sub_segments": [s],
                }

        if cur:
            units.append(cur)

        for i, u in enumerate(units):
            u["index"] = i + 1
            u["duration"] = round(u["end"] - u["start"], 3)
            u["clean_text"] = re.sub(r"[\n\r\t]", " ", u["text"]).strip()

        return units
