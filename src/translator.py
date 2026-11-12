"""
Show-agnostic subtitle translation with DeepSeek Closed-Loop ASR Quality Feedback.

Three passes, all automated:
  1. build_brief()        - DeepSeek skims the transcript and writes a structured "brief":
                            what the video is, who appears (merging ASR misspelled variants
                            into one canonical name), and domain slang/terms.
  2. translate_segments() - Translates the transcript in batches with DeepSeek KV prompt caching.
                            Whenever DeepSeek spots a line that looks like a broken ASR homophone,
                            garbled phrase, or music hallucination, it tags the tuple with
                            `{"redo": true, "reason": "...", "hint": "expected Chinese terms"}`.
  3. Closed-Loop Re-Subtitling (when `audio_path` is provided):
                            Flagged segments are automatically re-transcribed on their exact audio
                            window using the heavyweight `whisper-large-v3-mlx` model conditioned on
                            DeepSeek's brief + `hint`, and then re-translated by DeepSeek!
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from .llm_backends import LLMError

# Netflix guideline: 17 characters per second, 42 per line, 2 lines max.
CHARS_PER_SECOND = 17
MAX_SUBTITLE_CHARS = 84

BRIEF_SYSTEM = """You prepare material for a subtitle translator.
You will receive an automatically transcribed ({source}) transcript of a video you know nothing about.
Speech recognition often replaces names, dialect phrases, brand names, or English acronyms with similar-sounding
wrong ones, and the same person or term may appear spelled several different ways.

Read the transcript and reply with ONE JSON object and nothing else:
{{
  "summary": "1-3 sentences: what kind of video this is and what is going on",
  "tone": "register of the speech, e.g. casual banter, formal news, comedic",
  "people": [
    {{"as_heard": ["every spelling variant seen in the transcript"],
      "name": "the most likely correct name, in the source script",
      "romanized": "standard romanization / English form",
      "role": "who they seem to be (host, guest, contestant, ...)"}}
  ],
  "terms": [
    {{"source": "slang, idiom, wordplay, title or brand as it appears",
      "meaning": "what it really means here",
      "translation": "a good {target} rendering"}}
  ],
  "corrections": [
    {{"as_heard": "garbled phonetic homophone or acronym in transcript, e.g. 跟跟照照 or ito mct",
      "intended": "actual intended Chinese phrase or proper noun, e.g. 干干燥燥 or ITZY、NCT",
      "reason": "why, e.g. dialect homophone reduplication of 干爽 / K-pop group names"}}
  ]
}}
Only include entries you are reasonably confident about. Merge spelling variants of one name into one entry.
Use [] for any list you have nothing for."""

TRANSLATE_SYSTEM = """You are a professional subtitle translator and ASR quality auditor. Translate {source} into {target}.

The input is automatically transcribed speech, so it may contain recognition errors (wrong characters,
homophones, broken words, or background-music hallucinations). Use the surrounding lines and the brief
to work out what was really said; never translate phonetic nonsense literally.

Rules:
1. Names of people, places, shows and brands are never translated literally. Use the romanization from
   the brief (or the standard one) and keep every name identical throughout the video.
2. Slang, idioms, jokes and wordplay: convey the intended meaning and tone, not the literal words.
3. Subtitles must be quick to read and speak naturally. Each item has a "max_chars" limit: shorten by
   dropping filler rather than cramming.
4. Each item is one subtitle. Keep the ids; never merge, split, drop or reorder items.
   An item that continues the previous sentence should read naturally after it.
5. ASR FEEDBACK LOOP: If a source line is severely garbled by speech recognition (e.g. nonsensical
   homophones like "飞人鳞掌内" or "有妻分", unintelligible fragments, or obvious music/noise hallucination)
   such that the subtitler should re-listen to that audio slice with a stronger model, append a 3rd element
   to that item's output tuple:
   [id, "your best contextual English guess", {{"redo": true, "reason": "homophone_error|garbled|hallucination", "hint": "likely Chinese words/names being spoken here"}}]
   Only flag lines that are genuinely corrupted in the source text (typically <5-10% of lines).

Input items format: [[id, "source text", max_chars], ...]
Output format: Reply with ONLY a JSON array of tuples:
[[0, "English line zero"], [1, "Best guess for line one", {{"redo": true, "reason": "homophone_error", "hint": "神舟二十号、非人灵长类"}}], ...]
{brief_block}"""


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def extract_json(text: str):
    """Parses the first JSON value in an LLM reply, tolerating ```json fences and chatter around it."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in reply: {text[:200]!r}")


def _sample_for_brief(segments: list, char_budget: int) -> str:
    """The whole transcript if it fits; otherwise evenly spaced runs of consecutive lines."""
    lines = [s["text"].strip() for s in segments if s.get("text", "").strip()]
    if sum(len(l) + 1 for l in lines) <= char_budget:
        return "\n".join(lines)

    run = 20
    runs = [lines[i:i + run] for i in range(0, len(lines), run)]
    keep = max(1, int(char_budget / max(1, sum(len(l) + 1 for l in runs[0]))))
    step = max(1, len(runs) / keep)
    chosen = [runs[int(i * step)] for i in range(min(keep, len(runs)))]
    return "\n...\n".join("\n".join(r) for r in chosen)[:char_budget]


def format_brief(brief: Optional[dict]) -> str:
    """Renders the brief as compact text for the translation prompt."""
    if not brief:
        return ""
    out = ["", "Video brief (use it, but trust the transcript if it disagrees):"]
    if brief.get("summary"):
        out.append(f"- About: {brief['summary']}")
    if brief.get("tone"):
        out.append(f"- Tone: {brief['tone']}")
    for p in brief.get("people") or []:
        heard = "/".join(p.get("as_heard") or [p.get("name", "")])
        out.append(f"- Person: {heard} = {p.get('romanized') or p.get('name')} ({p.get('role', '')})")
    for t in brief.get("terms") or []:
        out.append(f"- Term: {t.get('source')} -> {t.get('translation')} ({t.get('meaning', '')})")
    for c in brief.get("corrections") or []:
        out.append(f"- ASR Repair: '{c.get('as_heard')}' was actually '{c.get('intended')}' ({c.get('reason', '')})")
    return "\n".join(out)


# ------------------------------------------------------------------------------
# pass 1: the brief
# ------------------------------------------------------------------------------
def build_brief(segments: list, backend, source_language: str = "Chinese",
                target_language: str = "English") -> dict:
    """Asks the model to skim the transcript and describe it. Returns {} if the reply is unusable."""
    sample = _sample_for_brief(segments, backend.context_chars)
    system = BRIEF_SYSTEM.format(source=source_language, target=target_language)
    try:
        brief = extract_json(backend.chat(system, sample))
        return brief if isinstance(brief, dict) else {}
    except (ValueError, LLMError) as e:
        print(f"⚠️  Could not build a brief ({e}). Continuing without one.")
        return {}


# ------------------------------------------------------------------------------
# pass 2: translation + DeepSeek ASR feedback extraction
# ------------------------------------------------------------------------------
def _max_chars(seg: dict) -> int:
    duration = max(0.0, seg["end"] - seg["start"])
    return int(min(MAX_SUBTITLE_CHARS, max(16, round(duration * CHARS_PER_SECOND))))


def _translate_items(backend, system: str, items: list, history: list) -> Tuple[Dict[int, str], Dict[int, dict]]:
    """
    Sends one batch; returns:
      - found: {id: translated_text}
      - flagged: {id: {"redo": True, "reason": ..., "hint": ...}}
    """
    request = {"translate": items, "context": history}
    reply = extract_json(backend.chat(system, json.dumps(request, ensure_ascii=False)))
    if isinstance(reply, dict):
        reply = next((v for v in reply.values() if isinstance(v, list)), [])

    wanted = {i[0] for i in items}
    found: Dict[int, str] = {}
    flagged: Dict[int, dict] = {}

    for row in reply:
        if isinstance(row, list) and len(row) >= 2 and isinstance(row[0], int) and isinstance(row[1], str):
            idx = row[0]
            if idx in wanted:
                found[idx] = row[1].strip()
                if len(row) >= 3 and isinstance(row[2], dict) and row[2].get("redo"):
                    flagged[idx] = row[2]
        elif isinstance(row, dict) and row.get("id") in wanted and isinstance(row.get("text"), str):
            idx = row["id"]
            found[idx] = row["text"].strip()
            if row.get("redo"):
                flagged[idx] = {
                    "redo": True,
                    "reason": row.get("reason", "flagged_by_llm"),
                    "hint": row.get("hint", ""),
                }
    return found, flagged


def translate_segments(
    segments: list,
    backend,
    brief: Optional[dict] = None,
    source_language: str = "Chinese",
    target_language: str = "English",
    batch_size: int = 80,
    history_lines: int = 4,
    audio_path: Optional[str] = None,
    enable_resubtitle_feedback: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list:
    """
    Fills seg["translated_text"] for every segment (in place) and returns the list.
    When `audio_path` is provided and `enable_resubtitle_feedback=True`, any segments
    flagged by DeepSeek (`"redo": true`) or with very low ASR logprob are automatically
    re-subtitled with the heavier `whisper-large-v3-mlx` model and re-translated!
    """
    system = TRANSLATE_SYSTEM.format(
        source=source_language, target=target_language, brief_block=format_brief(brief))

    # Apply brief-level phonetic corrections to source text in segments
    if brief and brief.get("corrections"):
        for item in brief["corrections"]:
            heard = item.get("as_heard", "").strip()
            intended = item.get("intended", "").strip()
            if heard and intended and heard != intended:
                for seg in segments:
                    if heard in seg.get("text", ""):
                        old_txt = seg["text"]
                        seg["text"] = seg["text"].replace(heard, intended)
                        seg["brief_corrected"] = True
                        print(f"  ✨ LLM Brief ASR Repair: '{heard}' -> '{intended}' in '{old_txt}'")

    total = len(segments)
    done = 0
    all_flagged: Dict[int, dict] = {}

    for start in range(0, total, batch_size):
        batch = list(range(start, min(total, start + batch_size)))
        items = [[i, segments[i]["text"], _max_chars(segments[i])] for i in batch]
        history = [
            f"{segments[i]['text']} -> {segments[i].get('translated_text', '')}"
            for i in range(max(0, start - history_lines), start)
            if segments[i].get("translated_text")
        ]

        found: Dict[int, str] = {}
        for attempt in range(2):
            remaining = [it for it in items if it[0] not in found]
            if not remaining:
                break
            try:
                batch_found, batch_flagged = _translate_items(backend, system, remaining, history)
                found.update(batch_found)
                all_flagged.update(batch_flagged)
            except (ValueError, LLMError) as e:
                print(f"⚠️  Batch starting at line {start + 1}: attempt {attempt + 1} failed ({e})")

        for i in batch:
            if i in found and found[i]:
                segments[i]["translated_text"] = found[i]
            else:
                segments[i]["translated_text"] = segments[i]["text"]
                segments[i]["translation_failed"] = True

        done += len(batch)
        if on_progress:
            on_progress(done, total)

    # Also include segments where primary Whisper had very low acoustic confidence (avg_logprob < -0.80)
    for i, seg in enumerate(segments):
        if i not in all_flagged and float(seg.get("avg_logprob", 0.0)) < -0.80:
            all_flagged[i] = {"redo": True, "reason": "low_asr_confidence", "hint": ""}

    # --------------------------------------------------------------------------
    # Pass 3: Closed-Loop Re-Subtitling & Re-Translation for Flagged Portions
    # --------------------------------------------------------------------------
    if enable_resubtitle_feedback and audio_path and all_flagged:
        print(f"\n🔄 DeepSeek & ASR Auditor flagged {len(all_flagged)}/{total} lines for targeted re-subtitling.")
        from .transcriber import resubtitle_flagged_segments

        updated_indices = resubtitle_flagged_segments(
            audio_path=audio_path,
            segments=segments,
            flagged_Map=all_flagged,
            brief=brief,
        )

        # Re-translate only the segments whose Chinese text actually changed (and weren't dropped)
        retranslate_ids = [
            i for i in updated_indices
            if segments[i].get("text", "").strip() and not segments[i].get("dropped_hallucination")
        ]
        if retranslate_ids:
            print(f"🧠 Sending {len(retranslate_ids)} re-subtitled lines back to DeepSeek for final translation...")
            items = [[i, segments[i]["text"], _max_chars(segments[i])] for i in retranslate_ids]
            try:
                re_found, _ = _translate_items(backend, system, items, history=[])
                for i, new_en in re_found.items():
                    if new_en:
                        old_en = segments[i].get("translated_text", "")
                        segments[i]["translated_text"] = new_en
                        print(f"  ✅ Line #{i + 1} EN updated: '{old_en}' -> '{new_en}'")
            except (ValueError, LLMError) as e:
                print(f"⚠️  Re-translation pass warning: {e}")

        # Remove segments that were proven to be pure BGM/silence hallucinations
        before_count = len(segments)
        segments[:] = [s for s in segments if not s.get("dropped_hallucination") and s.get("text", "").strip()]
        dropped_count = before_count - len(segments)
        if dropped_count:
            print(f"🧹 Removed {dropped_count} confirmed silence/BGM hallucination segments.")

    failed = sum(1 for s in segments if s.get("translation_failed"))
    if failed:
        print(f"⚠️  {failed} of {len(segments)} lines could not be translated and kept their source text.")
    return segments
