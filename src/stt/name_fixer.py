"""
name_fixer.py - Phonetic (Pinyin) Fuzzy Entity Corrector for Broadcast Chinese STT.

Resolves homophone and near-homophone misrecognitions of contestant names,
mentors, and specialized show terminology (e.g. 张凝凝 -> 张宁宁, 易立静 -> 易立竞).
Preserves surrounding punctuation and context.
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, List, Optional, Set, Tuple
import opencc
import pypinyin


def _normalize_syllable(py: str) -> str:
    """Normalizes retroflex sibilants, front/back nasals, and vowel mergers common in Southern/Taiwanese Mandarin."""
    s = py.lower()
    # Retroflex initials -> dental sibilants (zh->z, ch->c, sh->s)
    if s.startswith("zh"):
        s = "z" + s[2:]
    elif s.startswith("ch"):
        s = "c" + s[2:]
    elif s.startswith("sh"):
        s = "s" + s[2:]

    # Front/back nasal mergers
    if s.endswith("iang"):
        s = s[:-4] + "ian"
    elif s.endswith("uang"):
        s = s[:-4] + "uan"
    elif s.endswith("ang"):
        s = s[:-3] + "an"
    elif s.endswith("eng"):
        s = s[:-3] + "en"
    elif s.endswith("ing"):
        s = s[:-3] + "in"

    # Southern / Taiwan vowel relaxation (an <-> en, e.g. gan <-> gen)
    if s.endswith("an"):
        s = s[:-2] + "en"
    return s


def _normalize_dialect_pinyin(py_tuple: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(_normalize_syllable(p) for p in py_tuple)


class PhoneticNameFixer:
    """
    Scans Chinese text for phonetic homophones of registered roster names and terms,
    correcting them deterministically to their canonical forms.
    Supports Beijing Pinyin, polyphonic readings, and Southern/Taiwanese dialect normalization.
    """

    def __init__(self, names: List[str], terms: Optional[List[str]] = None):
        self.t2s = opencc.OpenCC("t2s")
        self.canonical_names: List[str] = [self.t2s.convert(n.strip()) for n in names if n.strip()]
        self.canonical_terms: List[str] = [self.t2s.convert(t.strip()) for t in (terms or []) if t.strip()]

        # Length index for fast sliding window search: len -> list of (raw_pinyin_tuple, norm_pinyin_tuple, canonical_string)
        self.targets_by_length: Dict[int, List[Tuple[Tuple[str, ...], Tuple[str, ...], str]]] = {}

        self._build_tables()

    def _build_tables(self) -> None:
        # Give names highest priority, then terms
        all_targets = list(self.canonical_names + self.canonical_terms)

        # Morphological reduplication expansion:
        # 2-character adjectives/terms like 干燥 -> 干干燥燥, 粉嫩 -> 粉粉嫩嫩 (AABB)
        expanded: List[str] = []
        for t in all_targets:
            expanded.append(t)
            if len(t) == 2 and re.match(r"^[\u4e00-\u9fa5]{2}$", t):
                aabb = t[0] * 2 + t[1] * 2
                expanded.append(aabb)

        # Deduplicate while preserving order
        seen = set()
        unique_targets = []
        for t in expanded:
            if t not in seen and len(t) >= 2:
                seen.add(t)
                unique_targets.append(t)

        for target in unique_targets:
            # Generate all possible pinyin variants for polyphonic characters (多音字)
            py_options = pypinyin.pinyin(target, heteronym=True, style=pypinyin.Style.NORMAL, errors="ignore")
            if not py_options:
                continue
            raw_tuples = list(itertools.product(*py_options))[:6]

            char_len = len(target)
            if char_len not in self.targets_by_length:
                self.targets_by_length[char_len] = []

            for raw_py in raw_tuples:
                norm_py = _normalize_dialect_pinyin(raw_py)
                self.targets_by_length[char_len].append((raw_py, norm_py, target))

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

                sub_py = tuple(pypinyin.lazy_pinyin(sub_chars, errors="ignore"))
                sub_norm = _normalize_dialect_pinyin(sub_py)

                for target_py, target_norm, target_str in self.targets_by_length[w]:
                    if sub_chars == target_str:
                        # Already correct, skip past it
                        matched = True
                        i += w
                        break

                    # Match strategy:
                    # 1. Exact Beijing Pinyin homophone match (any length >= 2)
                    # 2. Dialect-normalized match (for length >= 3 or AABB reduplications)
                    is_match = False
                    if sub_py == target_py:
                        is_match = True
                    elif w >= 3 and sub_norm == target_norm:
                        is_match = True

                    if is_match:
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
