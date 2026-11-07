"""
Interactive Review & Confirmation CLI for Actor Voice References.

Allows listening to candidate clips via macOS `afplay`, naming actors,
confirming/denying clusters, merging duplicate clusters, and exporting
a clean `confirmed_references/` bundle ready for zero-shot TTS cloning.
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional


def play_audio(wav_path: Path):
    """Plays an audio file via macOS native afplay."""
    if not wav_path.is_file():
        print(f"❌ File not found: {wav_path}")
        return
    print(f"▶️  Playing: {wav_path.name} (Ctrl+C to stop)...")
    try:
        subprocess.run(["afplay", str(wav_path)], check=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"⚠️ Playback error: {e}")


def interactive_review(
    manifest_path: str,
    output_confirmed_dir: str = "output/confirmed_references",
    auto_confirm: bool = False,
):
    m_path = Path(manifest_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {m_path}")

    base_dir = m_path.parent
    data = json.loads(m_path.read_text(encoding="utf-8"))
    speakers = data.get("speakers", {})
    total_speakers = len(speakers)

    print("\n" + "=" * 65)
    print("🎙️ ACTOR REFERENCE INTERACTIVE REVIEW & CONFIRMATION")
    print(f"📂 Discovered Clusters: {total_speakers}")
    print(f"📁 Manifest: {m_path.resolve()}")
    print("=" * 65)

    confirmed_actors: Dict[str, dict] = {}
    speaker_remap: Dict[str, str] = {}  # for merges

    out_ref_dir = Path(output_confirmed_dir)
    out_ref_dir.mkdir(parents=True, exist_ok=True)

    for spk_key, spk_data in speakers.items():
        spk_id = spk_data["speaker_id"]
        total_lines = spk_data["total_lines"]
        speaking_time = spk_data["total_speaking_time_sec"]
        samples = spk_data.get("samples", [])

        print("\n" + "-" * 60)
        print(f"🎭 [{spk_key}] — {total_lines} dialogue lines | {speaking_time:.1f}s total speech")
        print("-" * 60)

        for i, s in enumerate(samples, 1):
            print(f"  [{i}] ({s['duration']:.1f}s) \"{s['text']}\"")

        if auto_confirm:
            actor_name = f"Actor_{spk_id:02d}"
            primary = samples[0] if samples else None
            confirmed_actors[spk_key] = {
                "actor_name": actor_name,
                "status": "confirmed",
                "primary_sample": primary,
                "total_lines": total_lines,
                "total_speaking_time": speaking_time,
            }
            continue

        chosen_sample_idx = 0
        while True:
            prompt = (
                f"\nCommands for {spk_key}:\n"
                f"  p <num>           -> Play sample (e.g. 'p 1')\n"
                f"  c <Name>          -> Confirm as Actor Name (e.g. 'c Feng Lin' or just 'Feng Lin')\n"
                f"  d                 -> Deny / Discard (mark as noise/unwanted)\n"
                f"  m <other_id>      -> Merge into another Speaker (e.g. 'm Speaker_00' or 'm 0')\n"
                f"  s <num>           -> Set preferred primary voice sample (default: 1)\n"
                f"Action [c {spk_key}]: "
            )
            raw = input(prompt).strip()
            if not raw:
                raw = f"c {spk_key}"

            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "p":
                try:
                    s_idx = int(arg) - 1 if arg else 0
                    if 0 <= s_idx < len(samples):
                        sample_wav = base_dir / samples[s_idx]["wav_path"]
                        play_audio(sample_wav)
                    else:
                        print("❌ Invalid sample number.")
                except ValueError:
                    print("❌ Please specify sample number, e.g. 'p 1'.")
            elif cmd == "s":
                try:
                    s_idx = int(arg) - 1
                    if 0 <= s_idx < len(samples):
                        chosen_sample_idx = s_idx
                        print(f"✅ Set primary voice sample to #{s_idx + 1}")
                    else:
                        print("❌ Invalid sample number.")
                except ValueError:
                    print("❌ Please specify sample number, e.g. 's 2'.")
            elif cmd in ("d", "deny", "discard"):
                print(f"🗑️ Discarding {spk_key} as noise/unwanted.")
                speaker_remap[spk_key] = "DENIED"
                break
            elif cmd in ("m", "merge"):
                target = arg if arg.startswith("Speaker_") else f"Speaker_{int(arg):02d}"
                print(f"🔗 Merging {spk_key} into {target}.")
                speaker_remap[spk_key] = target
                break
            elif cmd == "c":
                actor_name = arg or spk_key
                primary = samples[chosen_sample_idx] if samples else None
                confirmed_actors[spk_key] = {
                    "actor_name": actor_name,
                    "status": "confirmed",
                    "primary_sample": primary,
                    "total_lines": total_lines,
                    "total_speaking_time": speaking_time,
                }
                print(f"✅ Confirmed {spk_key} as '{actor_name}'!")
                break
            else:
                # Treat any non-command string as the actor name
                actor_name = raw
                primary = samples[chosen_sample_idx] if samples else None
                confirmed_actors[spk_key] = {
                    "actor_name": actor_name,
                    "status": "confirmed",
                    "primary_sample": primary,
                    "total_lines": total_lines,
                    "total_speaking_time": speaking_time,
                }
                print(f"✅ Confirmed {spk_key} as '{actor_name}'!")
                break

    # Export confirmed references bundle
    print("\n" + "=" * 65)
    print(f"📦 Exporting confirmed actor voice bundle to: {out_ref_dir.resolve()}")

    export_manifest = {
        "confirmed_actors": {},
        "merges": speaker_remap,
    }

    for spk_key, info in confirmed_actors.items():
        name = info["actor_name"]
        slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name.lower()).strip("_") or spk_key.lower()
        primary = info.get("primary_sample")
        if primary:
            src_wav = base_dir / primary["wav_path"]
            dst_wav = out_ref_dir / f"{slug}_ref.wav"
            dst_txt = out_ref_dir / f"{slug}_ref.txt"

            shutil.copy2(src_wav, dst_wav)
            dst_txt.write_text(primary["text"], encoding="utf-8")

            export_manifest["confirmed_actors"][slug] = {
                "name": name,
                "original_cluster": spk_key,
                "ref_audio": str(dst_wav.resolve()),
                "ref_text": primary["text"],
                "duration": primary["duration"],
                "total_lines": info["total_lines"],
                "total_speaking_time_sec": info["total_speaking_time"],
            }
            print(f"  🌟 [{name}]: {dst_wav.name} ({primary['duration']:.2f}s) -> '{primary['text']}'")

    profile_json = out_ref_dir / "confirmed_actors.json"
    profile_json.write_text(json.dumps(export_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n🎉 Saved confirmed profiles -> {profile_json.resolve()}")
    return export_manifest
