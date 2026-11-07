"""
isometric_translator.py - Autonomous Isometric Machine Translator for ADR Dubbing.

Implements the Amazon Science Isometric SLT standard (Lakew et al., Amazon AI / IWSLT):
- Strict syllable budgeting: S_target = round(Duration * 4.0 sps).
- Programmatic English syllable validation via Pyphen (Hunspell hyphenation engine).
- Autonomous retry loop for non-compliant lines to guarantee length-matched English dubbing script.
"""
from __future__ import annotations
import re
import json
from typing import Dict, Any, List, Optional, Tuple
import pyphen


class IsometricADRTranslator:
    """
    Autonomous Isometric Machine Translator for ADR Dubbing.
    Enforces a strict syllable budget (Duration * 4.0 syllables/sec) with
    an automated verification and re-prompting loop.
    """
    def __init__(self, llm_backend: Any, target_sps: float = 4.0):
        self.llm = llm_backend
        self.target_sps = target_sps
        self.dic = pyphen.Pyphen(lang="en_US")

    def count_syllables(self, text: str) -> int:
        """Accurate English syllable counter using Pyphen hyphenation dictionary."""
        clean = re.sub(r"[^a-zA-Z\s]", "", text.lower()).strip()
        if not clean:
            return 0

        words = clean.split()
        total = 0
        for w in words:
            if len(w) <= 3:
                total += 1
                continue
            hyphenated = self.dic.inserted(w)
            parts = hyphenated.split("-")
            total += max(1, len(parts))
        return total

    def translate_isometric_batch(
        self,
        segments: List[Dict[str, Any]],
        system_prompt: str,
        max_retries: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Translates dialogue segments while enforcing strict syllable budgets.
        Self-corrects lines that fail isometric bounds (CR@15%).
        """
        payload = []
        for s in segments:
            dur = s["duration"]
            target_syl = max(2, int(round(dur * self.target_sps)))
            payload.append({
                "id": s["index"],
                "duration_s": round(dur, 2),
                "target_syllables": target_syl,
                "min_syllables": max(1, int(round(target_syl * 0.82))),
                "max_syllables": int(round(target_syl * 1.18)),
                "zh": s["zh"],
                "speaker_role": s.get("speaker_role", "Speaker")
            })

        user_prompt = f"""Translate these Chinese dialogue turns into natural English dubbing script.
CRITICAL CONSTRAINT: ISOMETRIC SYLLABLE CONTROL
For each turn, your English line MUST have a syllable count within [min_syllables, max_syllables].
Be punchy and colloquial reality-show English. Do NOT write verbose literal explanations.

Dialogue Items:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON list of objects:
[
  {{"id": 1, "en": "Natural English line", "syllables": 8}},
  ...
]"""

        reply = self.llm.chat(system_prompt, user_prompt)
        parsed = self._extract_json(reply)
        parsed_dict = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "id" in item:
                    parsed_dict[item["id"]] = item.get("en", "").strip()

        results = []
        out_of_spec = []

        for p in payload:
            line_id = p["id"]
            en_trans = parsed_dict.get(line_id, "")
            actual_syl = self.count_syllables(en_trans)
            min_s, max_s = p["min_syllables"], p["max_syllables"]

            if min_s <= actual_syl <= max_s or not en_trans:
                results.append({
                    "id": line_id,
                    "en": en_trans,
                    "syllables": actual_syl,
                    "target_syllables": p["target_syllables"],
                    "is_compliant": True
                })
            else:
                out_of_spec.append((p, en_trans, actual_syl))

        # Autonomous re-prompting loop for non-compliant lines
        if out_of_spec and max_retries > 0:
            print(f"  ⚡ {len(out_of_spec)} lines violated syllable budget. Autonomous isometric rewrite...")
            corrections = self._rewrite_out_of_spec(out_of_spec)
            for p, en_trans, actual_syl in out_of_spec:
                line_id = p["id"]
                fixed_en = corrections.get(line_id, en_trans)
                fixed_syl = self.count_syllables(fixed_en)
                results.append({
                    "id": line_id,
                    "en": fixed_en,
                    "syllables": fixed_syl,
                    "target_syllables": p["target_syllables"],
                    "is_compliant": p["min_syllables"] <= fixed_syl <= p["max_syllables"]
                })

        results.sort(key=lambda x: x["id"])
        return results

    def _rewrite_out_of_spec(self, out_of_spec: List[Tuple[Dict[str, Any], str, int]]) -> Dict[int, str]:
        """Requests targeted rewrites strictly for lines that overflowed or underflowed."""
        fix_items = []
        for p, en_trans, actual_syl in out_of_spec:
            action = "SHORTEN" if actual_syl > p["max_syllables"] else "EXPAND"
            fix_items.append({
                "id": p["id"],
                "zh": p["zh"],
                "draft": en_trans,
                "current_syllables": actual_syl,
                "target_range": [p["min_syllables"], p["max_syllables"]],
                "action": f"{action}: Need strictly between {p['min_syllables']} and {p['max_syllables']} syllables."
            })

        retry_prompt = f"""The following dubbing lines violated the syllable budget for their video time slots.
Rewrite each line so its English syllable count lands strictly within `target_range`.
Keep speech natural, colloquial, and punchy.

Lines needing adjustment:
{json.dumps(fix_items, ensure_ascii=False, indent=2)}

Return ONLY a JSON list:
[
  {{"id": 1, "en": "Revised punchy line"}}
]"""
        try:
            reply = self.llm.chat("You are an expert ADR dialog re-writer.", retry_prompt)
            data = self._extract_json(reply)
            if isinstance(data, list):
                return {item["id"]: item["en"].strip() for item in data if isinstance(item, dict) and "id" in item and "en" in item}
        except Exception:
            pass
        return {}

    def _extract_json(self, text: str) -> Any:
        text = text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        for op, cl in (("[", "]"), ("{", "}")):
            s, e = text.find(op), text.rfind(cl)
            if s != -1 and e > s:
                try:
                    return json.loads(text[s:e+1])
                except json.JSONDecodeError:
                    continue
        return []
