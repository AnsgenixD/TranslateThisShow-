#!/usr/bin/env python3
"""
local_vlm_arbiter.py - Local Apple Silicon VLM Active Speaker Arbiter.

Uses mlx-vlm and mlx-community/Qwen3-VL-4B-Instruct-4bit to arbitrate ambiguous
dialogue lines offline on Apple M4 GPU with zero API costs, rate limits, or network failures.

Workflow:
1. For each ambiguous or conflicting subtitle line:
   - Constructs a composite storyboard image:
     * Top Panel: Labeled reference face avatars for candidate actors.
     * Bottom Panel: 4 sequential video frames during the line with timestamps.
   - Prompts Qwen3-VL-4B to determine the physical active speaker and returns strict JSON.
2. Integrates seamlessly with existing arbitration cache:
   - Preserves all 18 previously cached Gemini arbitrations.
   - Arbitrates remaining ambiguous lines locally.
   - Writes upgraded decisions to output/diar_v2/mapped_subtitles_arbitrated.json.
"""

import os
import re
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"


def build_storyboard_image(
    video_path: str,
    start_sec: float,
    end_sec: float,
    candidate_ids: List[str],
    line_text: str,
    out_path: Path,
    actors_dir: str = "output/visual_actors"
) -> bool:
    """Creates a 1280-pixel wide composite storyboard with top candidate avatars and bottom sequential frames."""
    try:
        # 1. Extract 4 sequential frames from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        dur = max(0.2, end_sec - start_sec)
        time_offsets = [0.15, 0.40, 0.65, 0.90]
        video_frames = []

        for frac in time_offsets:
            t = start_sec + dur * frac
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ret, frame = cap.read()
            if not ret or frame is None:
                # Fallback blank frame if read fails
                f_img = np.zeros((180, 320, 3), dtype=np.uint8)
            else:
                f_img = cv2.resize(frame, (320, 180))
            # Draw timestamp overlay
            cv2.putText(f_img, f"{t:.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            video_frames.append(f_img)

        cap.release()

        # Stitch 4 video frames horizontally (1280 x 180)
        bottom_strip = np.hstack(video_frames)

        # 2. Build Top Panel with Candidate Avatars
        top_h = 160
        top_w = 1280
        top_panel = np.ones((top_h, top_w, 3), dtype=np.uint8) * 35  # Dark grey background

        # Header title
        header_text = f"Spoken Line: \"{line_text[:40]}\" | Candidates to compare against bottom video frames:"
        cv2.putText(top_panel, header_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        valid_cands = [c for c in candidate_ids if c.startswith("Person_")]
        if not valid_cands:
            cv2.putText(top_panel, "[No confirmed on-screen cast candidate - Check for off-screen narrator / guest]",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cand_box_w = 150
            total_cand_w = len(valid_cands) * cand_box_w
            start_x = max(20, (top_w - total_cand_w) // 2)

            for i, cid in enumerate(valid_cands):
                bx = start_x + i * cand_box_w
                avatar_path = Path(actors_dir) / cid / "avatar.jpg"
                if avatar_path.exists():
                    av = cv2.imread(str(avatar_path))
                    if av is not None:
                        av_resized = cv2.resize(av, (90, 90))
                        top_panel[40:130, bx+10:bx+100] = av_resized
                else:
                    cv2.rectangle(top_panel, (bx+10, 40), (bx+100, 130), (80, 80, 80), -1)

                cv2.putText(top_panel, cid, (bx+15, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # 3. Combine Top and Bottom Panels
        # Divider line
        divider = np.ones((4, top_w, 3), dtype=np.uint8) * 100
        composite = np.vstack([top_panel, divider, bottom_strip])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), composite)
        return True
    except Exception as e:
        print(f"⚠️ Error building storyboard for {out_path.name}: {e}")
        return False


def parse_vlm_json_response(raw_text: str, candidate_ids: List[str]) -> Optional[Dict[str, Any]]:
    """Extracts and validates structured JSON from the model's generated text."""
    # Look for ```json ... ``` or first { ... }
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text)
    if not match:
        match = re.search(r"(\{[\s\S]*?\})", raw_text)

    if not match:
        return None

    try:
        data = json.loads(match.group(1))
        spk = data.get("speaker_id", "")
        # Sanitize speaker_id
        if not spk:
            return None

        # Standardize known options
        conf = float(data.get("confidence", 0.75))
        is_on_screen = bool(data.get("is_on_screen", True))
        reason = str(data.get("reason", "VLM Active Speaker Detection"))

        return {
            "speaker_id": spk,
            "confidence": round(min(1.0, max(0.0, conf)), 2),
            "is_on_screen": is_on_screen,
            "reason": reason
        }
    except Exception:
        return None


def run_local_vlm_arbitration(
    raw_mapped_json: str = "output/diar_v2/mapped_subtitles_raw.json",
    video_path: str = "output/ep1_lower_30m.mp4",
    output_json: str = "output/diar_v2/mapped_subtitles_arbitrated.json",
    cache_json: str = "output/diar_v2/gemini_arbitration_cache.json",
    model_id: str = DEFAULT_MODEL_ID,
    confidence_threshold: float = 0.55
) -> List[Dict[str, Any]]:
    print("=" * 80)
    print("🧠 PHASE 5.5: LOCAL APPLE SILICON VLM ACTIVE SPEAKER ARBITER")
    print(f"🖥️ Model: {model_id} (Running on Apple M4 Metal GPU)")
    print("=" * 80)

    # 1. Load inputs
    with open(raw_mapped_json, "r", encoding="utf-8") as f:
        lines = json.load(f)

    cache_path = Path(cache_json)
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"📦 Loaded {len(cache)} existing cached decisions (preserving Gemini + prior arbitrations).")
        except Exception:
            cache = {}

    storyboard_dir = Path("output/diar_v2/vlm_storyboards")
    storyboard_dir.mkdir(parents=True, exist_ok=True)

    # 2. Filter lines that require arbitration
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

    print(f"🎯 Total lines: {len(lines)} | Lines needing local VLM arbitration: {len(lines_to_arbitrate)}")

    if lines_to_arbitrate:
        # Load local MLX-VLM model
        try:
            import mlx_vlm
            from mlx_vlm.utils import load_config
            from mlx_vlm.prompt_utils import apply_chat_template
            print(f"\n⚡ Initializing {model_id} on Apple Silicon Metal...")
            t_load = time.time()
            model, processor = mlx_vlm.load(model_id)
            config = load_config(model_id)
            print(f"✅ Local model loaded in {time.time()-t_load:.1f}s!")
        except ImportError:
            print("\n💡 [Optional Feature] Local VLM arbitration requires 'mlx-vlm'.")
            print("   To enable multi-panel storyboard speaker arbitration on Apple Silicon:")
            print("     pip install mlx-vlm")
            print(f"   Model: {model_id}")
            print("   Skipping VLM arbitration; preserving initial cluster attribution.\n")
            return lines
        except Exception as e:
            print(f"\n💡 [Notice] Could not load VLM model '{model_id}': {e}")
            print("   Skipping VLM arbitration; preserving initial cluster attribution.\n")
            return lines

        t0 = time.time()
        for count, l in enumerate(lines_to_arbitrate, 1):
            idx = l["index"]
            idx_str = str(idx)
            start_sec = l["start"]
            end_sec = l["end"]
            zh_text = l.get("zh", "")

            # Candidate gathering: combine ASD direct, cluster, and CAM++ top actor
            cands = []
            if l.get("direct_asd_actor"): cands.append(l["direct_asd_actor"])
            if l.get("cluster_actor"): cands.append(l["cluster_actor"])
            if l.get("campp_top_actor") and l.get("campp_top_score", 0.0) >= 0.65:
                cands.append(l["campp_top_actor"])
            candidates = list(dict.fromkeys([c for c in cands if c.startswith("Person_")]))

            # Build storyboard image
            sb_path = storyboard_dir / f"storyboard_{idx:04d}.jpg"
            if not sb_path.exists():
                ok = build_storyboard_image(video_path, start_sec, end_sec, candidates, zh_text, sb_path)
                if not ok:
                    print(f"  ⚠️ Could not build storyboard for Line #{idx}")
                    continue

            # Construct structured prompt
            cand_str = ", ".join([f"'{c}'" for c in candidates]) if candidates else "'Person_01'..'Person_12'"
            prompt_text = f"""You are an expert reality TV video editor performing active speaker attribution.
Examine this storyboard:
- TOP PANEL: Candidate cast reference faces.
- BOTTOM PANEL: 4 sequential video frames (left to right) during the spoken line: "{zh_text}" ({start_sec:.1f}s - {end_sec:.1f}s).

Tasks:
1. Examine the 4 bottom video frames closely. Look at who is actively moving their mouth or gesturing.
2. Compare the active person's face, hair, and clothing with the candidate avatars in the top panel.
3. If no candidate in the top panel is speaking, determine if it is an off-screen voiceover (NARRATOR or OFFSCREEN_UNKNOWN) or a guest speaker.

Allowed speaker IDs: {cand_str}, 'GUEST_SPEAKER_FEMALE', 'GUEST_SPEAKER_MALE', 'NARRATOR', 'OFFSCREEN_UNKNOWN'.

Respond ONLY with valid JSON in this exact schema:
```json
{{
  "speaker_id": "Person_XX",
  "confidence": 0.85,
  "is_on_screen": true,
  "reason": "1-sentence explanation citing lip motion and appearance match"
}}
```"""

            formatted_prompt = apply_chat_template(
                processor, config, prompt_text, num_images=1
            )

            try:
                t_inf = time.time()
                res = mlx_vlm.generate(
                    model,
                    processor,
                    prompt=formatted_prompt,
                    image=str(sb_path),
                    max_tokens=200,
                    verbose=False
                )
                dt = time.time() - t_inf
                raw_out = res.text if hasattr(res, "text") else str(res)
                parsed = parse_vlm_json_response(raw_out, candidates)

                if parsed:
                    parsed["line_index"] = idx
                    parsed["vlm_engine"] = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
                    cache[idx_str] = parsed
                    print(f"  [{count}/{len(lines_to_arbitrate)}] Line #{idx:04d} ({zh_text[:14]}...) [{dt:.1f}s] -> {parsed['speaker_id']} (conf={parsed['confidence']}): {parsed['reason']}")
                else:
                    # Fallback to direct ASD or cluster
                    fb_actor = candidates[0] if candidates else l.get("cluster_actor", "OFFSCREEN_UNKNOWN")
                    cache[idx_str] = {
                        "line_index": idx,
                        "speaker_id": fb_actor,
                        "confidence": 0.60,
                        "is_on_screen": True,
                        "reason": "VLM parsing fallback to candidate",
                        "vlm_engine": "fallback"
                    }
                    print(f"  [{count}/{len(lines_to_arbitrate)}] Line #{idx:04d} ({zh_text[:14]}...) -> [Fallback] {fb_actor}")

                # Save cache periodically
                if count % 10 == 0 or count == len(lines_to_arbitrate):
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"  ⚠️ Line #{idx} VLM inference error: {e}")

        print(f"\n✅ Local VLM arbitration completed in {time.time()-t0:.1f}s!")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    # 3. Apply cached arbitration decisions to dialogue manifest
    arbitrated_lines = []
    upgraded_count = 0

    for l in lines:
        line_copy = dict(l)
        idx_str = str(l["index"])

        if idx_str in cache and "speaker_id" in cache[idx_str]:
            arb = cache[idx_str]
            new_actor = arb["speaker_id"]

            line_copy["arbitrated"] = True
            line_copy["vlm_speaker"] = new_actor
            line_copy["vlm_confidence"] = arb.get("confidence", 0.85)
            line_copy["vlm_reason"] = arb.get("reason", "")
            line_copy["vlm_engine"] = arb.get("vlm_engine", "gemini_multimodal")

            line_copy["actor_id"] = new_actor
            line_copy["actor_name"] = new_actor
            line_copy["confidence"] = arb.get("confidence", 0.85)
            line_copy["match_type"] = "vlm_active_speaker_arbiter"
            upgraded_count += 1

        arbitrated_lines.append(line_copy)

    print(f"🚀 Upgraded {upgraded_count} dialogue lines with multimodal visual reasoning!")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(arbitrated_lines, f, ensure_ascii=False, indent=2)

    return arbitrated_lines


if __name__ == "__main__":
    run_local_vlm_arbitration()
