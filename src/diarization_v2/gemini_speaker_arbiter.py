#!/usr/bin/env python3
"""
gemini_speaker_arbiter.py - Resilient Multimodal Active Speaker Arbiter with Free-Tier Protection.

Features:
1. Dynamic Model Fallback Chain: Rotates across video-capable models
   ['gemini-3.5-flash', 'gemini-3.6-flash'] upon 429 RESOURCE_EXHAUSTED or 503 UNAVAILABLE.
2. Graceful Quota Boundary: If daily free-tier quota is reached across models, cleanly saves
   all cached decisions and falls back to audio-visual attribution without crashing.
3. Adaptive Pacing: Sleeps 3.0s between requests to stay strictly within Free Tier RPM limits (<= 15 RPM).
4. Strict Timeout: Uses 15,000ms HTTP timeout to prevent API hangs.
5. Persistent JSON Cache: Resumes automatically without re-querying previously resolved lines.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_FALLBACK_POOL = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]


def extract_video_snippet(video_path: str, start_sec: float, end_sec: float, out_path: Path) -> bool:
    """Extracts a tiny 320p H.264 video snippet (typically < 100 KB) for fast API upload."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0.0, start_sec - 0.2)),
        "-to", str(end_sec + 0.2),
        "-i", video_path,
        "-vf", "scale=360:-2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-c:a", "aac", "-b:a", "32k",
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path.exists() and out_path.stat().st_size > 0


def query_gemini_with_fallback(
    client: genai.Client,
    vfile: Any,
    prompt: str,
    active_model_idx: int = 0
) -> Tuple[Dict[str, Any], int]:
    """Queries Gemini with automatic fallback to alternate models upon 429/503 errors."""
    num_models = len(MODEL_FALLBACK_POOL)
    current_idx = active_model_idx

    for attempt in range(num_models):
        model_name = MODEL_FALLBACK_POOL[current_idx % num_models]
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[vfile, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            result = json.loads(resp.text)
            return result, (current_idx % num_models)
        except Exception as e:
            err_str = str(e)
            is_quota = "RESOURCE_EXHAUSTED" in err_str or "429" in err_str
            is_capacity = "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str

            if is_quota or is_capacity:
                reason = "quota_exhausted" if is_quota else "high_demand"
                print(f"    ⚠️ Model {model_name} hit {reason}. Backing off to next model...")
                current_idx += 1
                time.sleep(2.0)
            else:
                print(f"    ⚠️ Model {model_name} error: {err_str[:80]}. Trying next model...")
                current_idx += 1
                time.sleep(2.0)

    raise RuntimeError("ALL_MODELS_EXHAUSTED")


def arbitrate_single_line_resilient(
    client: genai.Client,
    video_path: str,
    line: Dict[str, Any],
    candidate_ids: List[str],
    cache_dir: Path,
    active_model_idx: int = 0
) -> Tuple[Dict[str, Any], int]:
    idx = line["index"]
    start_sec = line["start"]
    end_sec = line["end"]
    line_text = line.get("zh", line.get("text", ""))

    tmp_clip = cache_dir / f"clip_{idx:04d}_{int(start_sec*10)}.mp4"
    if not extract_video_snippet(video_path, start_sec, end_sec, tmp_clip):
        return {"error": "Failed to extract video snippet"}, active_model_idx

    # Upload snippet with retry
    vfile = None
    for attempt in range(3):
        try:
            vfile = client.files.upload(file=str(tmp_clip))
            while vfile.state.name == "PROCESSING":
                time.sleep(0.5)
                vfile = client.files.get(name=vfile.name)
            break
        except Exception as e:
            if attempt == 2:
                if tmp_clip.exists(): tmp_clip.unlink()
                return {"error": f"Upload failed: {e}"}, active_model_idx
            time.sleep(2.0)

    c_str = ", ".join(candidate_ids) if candidate_ids else "Person_01 through Person_12"
    prompt = f"""You are an expert reality TV broadcast editor performing speaker identification.
Watch this short video clip.
Spoken line: "{line_text}" (timestamp: {start_sec:.1f}s - {end_sec:.1f}s).
Candidate cast members: {c_str}.
Open-set options: 'GUEST_SPEAKER_FEMALE', 'GUEST_SPEAKER_MALE', 'NARRATOR', 'OFFSCREEN_UNKNOWN'.

Task:
Determine who is physically speaking this line.
Return a JSON object with:
- 'speaker_id': one of the candidate cast members or open-set options.
- 'confidence': float between 0.0 and 1.0.
- 'is_on_screen': bool (true if speaker is visible moving lips, false if offscreen/voiceover).
- 'reason': brief 1-sentence explanation citing lip motion, glasses, clothing, or scene context."""

    try:
        result, new_model_idx = query_gemini_with_fallback(client, vfile, prompt, active_model_idx)
        result["line_index"] = idx
    except Exception as e:
        result = {"error": str(e), "line_index": idx}
        new_model_idx = active_model_idx
    finally:
        try: client.files.delete(name=vfile.name)
        except Exception: pass
        if tmp_clip.exists(): tmp_clip.unlink()

    return result, new_model_idx


def run_gemini_arbitration(
    raw_mapped_json: str = "output/diar_v2/mapped_subtitles_raw.json",
    video_path: str = "output/ep1_lower_30m.mp4",
    output_json: str = "output/diar_v2/mapped_subtitles_arbitrated.json",
    cache_json: str = "output/diar_v2/gemini_arbitration_cache.json",
    pacing_delay_sec: float = 3.0,
    confidence_threshold: float = 0.50
) -> List[Dict[str, Any]]:
    print("=" * 75)
    print("🧠 PHASE 5.5: RESILIENT GEMINI ACTIVE SPEAKER ARBITER (WITH MODEL FALLBACK)")
    print("=" * 75)

    with open(raw_mapped_json, "r", encoding="utf-8") as f:
        lines = json.load(f)

    cache_path = Path(cache_json)
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"📦 Loaded {len(cache)} cached Gemini arbitration decisions.")
        except Exception:
            cache = {}

    clip_cache_dir = Path("output/diar_v2/gemini_clips")
    clip_cache_dir.mkdir(parents=True, exist_ok=True)

    # Focus on genuine cast conflicts and low-confidence lines
    lines_to_arbitrate = []
    for l in lines:
        idx = str(l["index"])
        conf = l.get("confidence", 0.5)
        d_act = l.get("direct_asd_actor")
        c_act = l.get("cluster_actor")

        is_conflict = bool(d_act and c_act and d_act != c_act and d_act.startswith("Person_") and c_act.startswith("Person_"))
        is_low_conf = (conf < confidence_threshold)

        if is_conflict or is_low_conf:
            if idx not in cache:
                lines_to_arbitrate.append(l)

    print(f"🎯 Total lines: {len(lines)} | Needing Gemini arbitration: {len(lines_to_arbitrate)} (Already cached: {len(cache)})")
    print(f"🔄 Active Model Fallback Pool: {MODEL_FALLBACK_POOL}")
    print(f"⏱️ Adaptive Pacing Delay: {pacing_delay_sec}s between calls (ensures Free Tier compliance)")

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=15000)  # 15s timeout in ms
    )
    active_model_idx = 0
    quota_exhausted_flag = False

    if lines_to_arbitrate:
        t0 = time.time()
        for count, l in enumerate(lines_to_arbitrate, 1):
            if quota_exhausted_flag:
                print(f"⚠️ Free tier daily quota reached across all models. Skipping remaining {len(lines_to_arbitrate) - count + 1} lines and preserving classical attribution.")
                break

            candidates = []
            if l.get("direct_asd_actor"): candidates.append(l["direct_asd_actor"])
            if l.get("cluster_actor"): candidates.append(l["cluster_actor"])
            candidates = list(dict.fromkeys(candidates))

            res, active_model_idx = arbitrate_single_line_resilient(
                client=client,
                video_path=video_path,
                line=l,
                candidate_ids=candidates,
                cache_dir=clip_cache_dir,
                active_model_idx=active_model_idx
            )

            idx_str = str(l["index"])
            if "error" not in res:
                cache[idx_str] = res
                print(f"  [{count}/{len(lines_to_arbitrate)}] Line #{l['index']} ({l['zh'][:14]}...) [Model: {MODEL_FALLBACK_POOL[active_model_idx]}] -> {res.get('speaker_id')} (conf={res.get('confidence')}): {res.get('reason')}")
                # Save cache after every successful arbitration
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
            else:
                err_msg = res["error"]
                if "ALL_MODELS_EXHAUSTED" in err_msg:
                    quota_exhausted_flag = True
                print(f"  ⚠️ Line #{l['index']} skipped: {err_msg}")

            # Pacing delay
            time.sleep(pacing_delay_sec)

        print(f"\n✅ Completed Gemini video arbitration pass in {time.time()-t0:.1f}s.")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    # Apply all cached arbitration decisions to dialogue manifest
    arbitrated_lines = []
    upgraded_count = 0

    for l in lines:
        line_copy = dict(l)
        idx_str = str(l["index"])

        if idx_str in cache and "speaker_id" in cache[idx_str]:
            arb = cache[idx_str]
            new_actor = arb["speaker_id"]

            line_copy["gemini_arbitrated"] = True
            line_copy["gemini_speaker"] = new_actor
            line_copy["gemini_confidence"] = arb.get("confidence", 0.9)
            line_copy["gemini_reason"] = arb.get("reason", "")
            line_copy["gemini_is_on_screen"] = arb.get("is_on_screen", True)

            line_copy["actor_id"] = new_actor
            line_copy["actor_name"] = new_actor
            line_copy["confidence"] = arb.get("confidence", 0.9)
            line_copy["match_type"] = "gemini_multimodal_arbiter"
            upgraded_count += 1

        arbitrated_lines.append(line_copy)

    print(f"🚀 Upgraded {upgraded_count} dialogue lines with Gemini visual reasoning!")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(arbitrated_lines, f, ensure_ascii=False, indent=2)

    return arbitrated_lines


if __name__ == "__main__":
    run_gemini_arbitration()
