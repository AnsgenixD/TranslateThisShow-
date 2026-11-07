"""
name_fixer.py - Phonetic (Pinyin) Fuzzy Entity Corrector for Broadcast Chinese STT.

Resolves homophone and near-homophone misrecognitions of contestant names,
mentors, and specialized show terminology (e.g. 张凝凝 -> 张宁宁, 易立静 -> 易立竞).
Preserves surrounding punctuation and context.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
import opencc
import pypinyin


class PhoneticNameFixer:
    """
    Scans Chinese text for phonetic homophones of registered roster names and terms,
    correcting them deterministically to their canonical forms.
    """

    def __init__(self, names: List[str], terms: Optional[List[str]] = None):
        self.t2s = opencc.OpenCC("t2s")
        self.canonical_names: List[str] = [self.t2s.convert(n.strip()) for n in names if n.strip()]
        self.canonical_terms: List[str] = [self.t2s.convert(t.strip()) for t in (terms or []) if t.strip()]

        # Build pinyin lookup tables
        # Map: tuple(pinyin_tokens) -> canonical_string
        self.pinyin_to_canonical: Dict[Tuple[str, ...], str] = {}
        # Length index for fast sliding window search
        self.targets_by_length: Dict[int, List[Tuple[Tuple[str, ...], str]]] = {}

        self._build_tables()

    def _build_tables(self) -> None:
        # Give names highest priority, then terms
        all_targets = self.canonical_names + self.canonical_terms
        # Deduplicate while preserving order
        seen = set()
        unique_targets = []
        for t in all_targets:
            if t not in seen and len(t) >= 2:
                seen.add(t)
                unique_targets.append(t)

        for target in unique_targets:
            py = tuple(pypinyin.lazy_pinyin(target, errors="ignore"))
            if not py:
                continue
            self.pinyin_to_canonical[py] = target
            char_len = len(target)
            if char_len not in self.targets_by_length:
                self.targets_by_length[char_len] = []
            self.targets_by_length[char_len].append((py, target))

    def fix_text(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Corrects phonetic misspellings in the input text.
        Returns (corrected_text, list_of_replacements).
        """
        if not text:
            return "", []

        # Convert traditional characters to simplified for uniform matching
        s_text = self.t2s.convert(text)
        chars = list(s_text)
        n_chars = len(chars)
        replacements: List[Dict[str, str]] = []

        # Sort window sizes descending (e.g. 4-char names/terms first, then 3, then 2)
        window_sizes = sorted(self.targets_by_length.keys(), reverse=True)

        i = 0
        while i < n_chars:
            matched = False
            for w in window_sizes:
                if i + w > n_chars:
                    continue

                sub_chars = "".join(chars[i:i + w])
                # Only check if all characters in the window are CJK ideographs
                if not re.match(r"^[\u4e00-\u9fa5]+$", sub_chars):
                    continue

                # Check if it is already identical to the canonical target
                for target_py, target_str in self.targets_by_length[w]:
                    if sub_chars == target_str:
                        # Already correct, skip past it
                        matched = True
                        i += w
                        break

                    # Compare pinyin representation
                    sub_py = tuple(pypinyin.lazy_pinyin(sub_chars, errors="ignore"))
                    if sub_py == target_py:
                        # Exact homophone match found!
                        chars[i:i + w] = list(target_str)
                        replacements.append({
                            "original": sub_chars,
                            "corrected": target_str,
                            "position": str(i),
                        })
                        matched = True
                        i += len(target_str)
                        break

                if matched:
                    break

            if not matched:
                i += 1

        corrected_text = "".join(chars)
        return corrected_text, replacements

    def fix_segments(self, segments: List[dict], text_key: str = "text") -> List[dict]:
        """Applies correction in-place or copies across a list of subtitle segment dictionaries."""
        for seg in segments:
            orig = seg.get(text_key, "")
            if orig:
                fixed, reps = self.fix_text(orig)
                seg[text_key] = fixed
                if reps:
                    seg["name_fixes"] = reps
        return segments
