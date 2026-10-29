"""
Show-agnostic subtitle translation.

Two passes, both done by an LLM, so the user never has to write a glossary by hand:
  1. build_brief()        - the model skims the whole transcript and writes a short "brief":
                            what the video is, who appears (with the ASR's misspelled variants
                            merged into one correct name), and slang/terms that need care.
  2. translate_segments() - the transcript is translated in small batches. Every batch gets the
                            brief plus a few already-translated lines, so names and tone stay
                            consistent across the whole video.

Nothing in here is specific to any show or language pair.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from .llm_backends import LLMError

# Netflix's English guideline for adult programmes is 17 characters per second, 42 per line, 2 lines.
CHARS_PER_SECOND = 17
MAX_SUBTITLE_CHARS = 84

BRIEF_SYSTEM = """You prepare material for a subtitle translator.
You will receive an automatically transcribed ({source}) transcript of a video you know nothing about.
Speech recognition often replaces names and rare words with similar-sounding wrong ones, and the same
person may appear spelled several different ways.

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
  ]
}}
Only include entries you are reasonably confident about. Merge spelling variants of one name into one entry.
Use [] for any list you have nothing for."""

TRANSLATE_SYSTEM = """You are a professional subtitle translator. Translate {source} into {target}.

The input is automatically transcribed speech, so it may contain recognition errors (wrong characters,
homophones, broken words). Use the surrounding lines and the brief to work out what was really said;
never translate nonsense literally.

Rules:
1. Names of people, places, shows and brands are never translated literally. Use the romanization from
   the brief (or the standard one) and keep every name identical throughout the video.
2. Slang, idioms, jokes and wordplay: convey the intended meaning and tone, not the literal words.
3. Subtitles must be quick to read. Each item has a "max_chars" limit: shorten by dropping filler
   rather than cramming. Never exceed it.
4. Each item is exactly one subtitle. Keep the ids; never merge, split, drop or reorder items.
   An item that continues the previous sentence should read naturally after it.
5. No notes or commentary.

Reply with ONLY a JSON array: [{{"id": <int>, "text": "<translation>"}}, ...] covering every item in "translate".
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
    lines = [s["text"].strip() for s in segments if s["text"].strip()]
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
# pass 2: translation
# ------------------------------------------------------------------------------
def _max_chars(seg: dict) -> int:
    duration = max(0.0, seg["end"] - seg["start"])
    return int(min(MAX_SUBTITLE_CHARS, max(12, round(duration * CHARS_PER_SECOND))))


def _translate_items(backend, system: str, items: list, history: list) -> dict:
    """Sends one batch; returns {id: translated_text} for whatever came back valid."""
    request = {"previous_lines": history, "translate": items}
    reply = extract_json(backend.chat(system, json.dumps(request, ensure_ascii=False)))
    if isinstance(reply, dict):  # some models wrap the array: {"translations": [...]}
        reply = next((v for v in reply.values() if isinstance(v, list)), [])
    wanted = {i["id"] for i in items}
    found = {}
    for row in reply:
        if isinstance(row, dict) and row.get("id") in wanted and isinstance(row.get("text"), str):
            found[row["id"]] = row["text"].strip()
    return found


def translate_segments(
    segments: list,
    backend,
    brief: Optional[dict] = None,
    source_language: str = "Chinese",
    target_language: str = "English",
    batch_size: int = 25,
    history_lines: int = 3,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list:
    """
    Fills seg["translated_text"] for every segment (in place) and returns the list.
    Lines the model fails to return after one retry keep their source text and get
    seg["translation_failed"] = True, so a bad batch never silently loses a subtitle.
    """
    system = TRANSLATE_SYSTEM.format(
        source=source_language, target=target_language, brief_block=format_brief(brief))

    total = len(segments)
    done = 0
    for start in range(0, total, batch_size):
        batch = list(range(start, min(total, start + batch_size)))
        items = [{"id": i, "text": segments[i]["text"], "max_chars": _max_chars(segments[i])} for i in batch]
        history = [{"source": segments[i]["text"], "translation": segments[i].get("translated_text", "")}
                   for i in range(max(0, start - history_lines), start)]

        found = {}
        for attempt in range(2):
            remaining = [it for it in items if it["id"] not in found]
            if not remaining:
                break
            try:
                found.update(_translate_items(backend, system, remaining, history))
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

    failed = sum(1 for s in segments if s.get("translation_failed"))
    if failed:
        print(f"⚠️  {failed} of {total} lines could not be translated and kept their source text.")
    return segments
