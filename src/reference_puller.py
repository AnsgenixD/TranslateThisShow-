"""
Interactive Speaker Reference Puller & Actor Speech Classifier.

Extracts vocal embeddings from isolated Demucs vocals using WavLM (microsoft/wavlm-base-plus-sv),
clusters them into distinct actor vocal profiles (with optional user-specified speaker count),
and saves each actor's top candidate speech samples with transcripts into a structured folder
for user review, confirmation, or denial.
"""
from __future__ import annotations

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


def parse_srt(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\d+\n|\Z)"
    )
    entries = []
    for m in pattern.finditer(text):
        idx, start_str, end_str, content = m.groups()
        h, mn, s_ms = start_str.split(":")
        s, ms = s_ms.split(",")
        start_sec = int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        h, mn, s_ms = end_str.split(":")
        s, ms = s_ms.split(",")
        end_sec = int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        entries.append({
            "index": int(idx),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(end_sec - start_sec, 3),
            "start_str": start_str,
            "end_str": end_str,
            "text": content.strip().replace("\n", " "),
        })
    return entries


def extract_vocal_embeddings(
    vocals_path: str,
    entries: List[dict],
    device: Optional[torch.device] = None,
) -> Tuple[List[Optional[np.ndarray]], List[int], np.ndarray, int]:
    """
    Extracts normalized WavLM embeddings from the isolated vocal stem.
    Filters out background noise, sub-second murmurs, or silent gaps.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n🎧 Loading vocal stem: {vocals_path}")
    audio_full, sr = sf.read(vocals_path, dtype="float32")
    if audio_full.ndim > 1:
        audio_mono = np.mean(audio_full, axis=1)
    else:
        audio_mono = audio_full

    print(f"🧠 Loading WavLM speaker verification model on {device}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
    model.eval()

    embeddings: List[Optional[np.ndarray]] = []
    valid_indices: List[int] = []

    print(f"📊 Extracting speaker embeddings across {len(entries)} dialogue segments...")
    t0 = time.time()

    for i, e in enumerate(entries):
        s_sample = int(e["start"] * sr)
        e_sample = int(e["end"] * sr)
        segment = audio_mono[s_sample:e_sample]

        # Ignore micro-clips (<0.45s) or silent segments (RMS < 0.007)
        if len(segment) < int(0.45 * sr):
            embeddings.append(None)
            continue

        rms = float(np.sqrt(np.mean(segment ** 2)))
        if rms < 0.007:
            embeddings.append(None)
            continue

        if sr != 16000:
            seg_16k = librosa.resample(segment, orig_sr=sr, target_sr=16000)
        else:
            seg_16k = segment

        inputs = feature_extractor(seg_16k, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            emb = model(input_values).embeddings
            emb_norm = emb / torch.norm(emb, dim=-1, keepdim=True)
            emb_vec = emb_norm.squeeze().cpu().numpy()

        embeddings.append(emb_vec)
        valid_indices.append(i)

        if (i + 1) % 150 == 0 or (i + 1) == len(entries):
            print(f"  -> Extracted {i + 1}/{len(entries)} embeddings ({time.time() - t0:.1f}s)")

    print(f"✅ Extracted {len(valid_indices)} high-confidence vocal embeddings in {time.time() - t0:.1f}s")
    return embeddings, valid_indices, audio_mono, sr


def cluster_vocal_profiles(
    embeddings: List[Optional[np.ndarray]],
    valid_indices: List[int],
    entries: List[dict],
    num_speakers: Optional[int] = None,
    distance_threshold: float = 0.44,
) -> Tuple[Dict[int, int], int]:
    """
    Clusters speaker embeddings.
    If `num_speakers` is explicitly specified by the user, fits exact `n_clusters`.
    Otherwise, evaluates silhouette scores or applies a tight cosine distance threshold.
    """
    valid_embs = np.array([embeddings[i] for i in valid_indices])
    n_samples = len(valid_embs)
    if n_samples < 2:
        return {i: 0 for i in range(len(entries))}, 1

    if num_speakers is not None and num_speakers > 0:
        target_k = min(num_speakers, n_samples)
        print(f"\n🎯 Clustering with user-specified actor count: {target_k} actors...")
        clustering = AgglomerativeClustering(
            n_clusters=target_k,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(valid_embs)
        final_k = target_k
    else:
        print("\n🔍 Auto-detecting optimal actor cluster count...")
        # Search silhouette scores for k in [3, min(14, n_samples-1)]
        best_k = None
        best_score = -1.0
        max_k = min(14, n_samples - 1)

        if max_k >= 3:
            for k in range(3, max_k + 1):
                try:
                    c = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
                    lbls = c.fit_predict(valid_embs)
                    if len(set(lbls)) > 1:
                        score = silhouette_score(valid_embs, lbls, metric="cosine")
                        if score > best_score:
                            best_score = score
                            best_k = k
                except Exception:
                    pass

        if best_k and best_score >= 0.32:
            print(f"🎯 Optimal silhouette cluster count: {best_k} actors (score={best_score:.3f})")
            clustering = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average")
            labels = clustering.fit_predict(valid_embs)
            final_k = best_k
        else:
            print(f"🎯 Clustering with fine-grained cosine threshold={distance_threshold}...")
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(valid_embs)
            final_k = len(set(labels))

    print(f"✅ Discovered {final_k} distinct speaker clusters across dialogue lines.")

    # Map labels back to all entries
    speaker_map: Dict[int, int] = {}
    for pos, entry_idx in enumerate(valid_indices):
        speaker_map[entry_idx] = int(labels[pos])

    # Temporal context filling for skipped micro-lines
    for i in range(len(entries)):
        if i not in speaker_map:
            prev_spk = next((speaker_map[p] for p in range(i - 1, -1, -1) if p in speaker_map), 0)
            next_spk = next((speaker_map[n] for n in range(i + 1, len(entries)) if n in speaker_map), prev_spk)
            speaker_map[i] = prev_spk if prev_spk == next_spk else prev_spk

    # Temporal de-jitter smoothing: remove single-line ping-pongs
    for i in range(1, len(entries) - 1):
        if speaker_map[i - 1] == speaker_map[i + 1] and speaker_map[i] != speaker_map[i - 1]:
            speaker_map[i] = speaker_map[i - 1]

    return speaker_map, final_k


def save_actor_speech_samples(
    audio_mono: np.ndarray,
    sr: int,
    entries: List[dict],
    speaker_map: Dict[int, int],
    num_speakers: int,
    output_dir: str = "output/speaker_candidates",
    samples_per_speaker: int = 5,
    target_sr: int = 24000,
) -> Dict[str, any]:
    """
    Saves candidate speech audio files and transcripts into subfolders:
      output_dir/
        Speaker_00/
          sample_01.wav, sample_01.txt, sample_02.wav...
          summary.json
        Speaker_01/
        ...
        review_manifest.json
        review_player.html (Interactive audio player)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_lines": len(entries),
        "total_speakers": num_speakers,
        "speakers": {},
    }

    print(f"\n📂 Saving actor speech folders to: {out_path.resolve()}")

    for spk_id in range(num_speakers):
        spk_indices = [i for i, s in speaker_map.items() if s == spk_id]
        if not spk_indices:
            continue

        spk_dir = out_path / f"Speaker_{spk_id:02d}"
        spk_dir.mkdir(parents=True, exist_ok=True)

        # Score candidate lines: prefer duration in [2.2s, 5.0s], clean RMS, and non-empty text
        candidates = []
        total_speaking_time = 0.0

        for idx in spk_indices:
            e = entries[idx]
            dur = e["duration"]
            total_speaking_time += dur
            text = e["text"]
            s_sample = int(e["start"] * sr)
            e_sample = int(e["end"] * sr)
            clip = audio_mono[s_sample:e_sample]
            if len(clip) == 0:
                continue

            rms = float(np.sqrt(np.mean(clip ** 2)))
            clean_chars = len(re.sub(r"[^\w\u4e00-\u9fff]", "", text))

            # Quality scoring
            if 2.2 <= dur <= 5.0:
                dur_factor = 1.0
            elif 1.4 <= dur <= 7.0:
                dur_factor = 0.7
            else:
                dur_factor = 0.3

            char_factor = 1.0 if clean_chars >= 6 else (0.5 if clean_chars >= 2 else 0.1)
            score = rms * dur_factor * char_factor
            candidates.append({
                "score": score,
                "index": idx,
                "entry": e,
                "clip": clip,
                "rms": rms,
            })

        # Sort descending by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:samples_per_speaker]

        samples_meta = []
        for rank, cand in enumerate(top_candidates, 1):
            e = cand["entry"]
            clip = cand["clip"]

            # Normalize peak to -1.0 dBFS
            peak = np.max(np.abs(clip)) + 1e-6
            norm_clip = clip / peak * 0.90

            wav_name = f"sample_{rank:02d}.wav"
            txt_name = f"sample_{rank:02d}.txt"
            wav_path = spk_dir / wav_name
            txt_path = spk_dir / txt_name

            # Resample to target 24kHz mono
            if sr != target_sr:
                clip_24k = librosa.resample(norm_clip, orig_sr=sr, target_sr=target_sr)
                sf.write(str(wav_path), clip_24k, target_sr)
            else:
                sf.write(str(wav_path), norm_clip, target_sr)

            txt_content = (
                f"Line #{e['index']} [{e['start_str']} --> {e['end_str']}] (Duration: {e['duration']:.2f}s, RMS: {cand['rms']:.4f})\n"
                f"{e['text']}\n"
            )
            txt_path.write_text(txt_content, encoding="utf-8")

            samples_meta.append({
                "rank": rank,
                "line_index": e["index"],
                "start": e["start"],
                "end": e["end"],
                "duration": e["duration"],
                "text": e["text"],
                "wav_path": str(wav_path.relative_to(out_path)),
                "txt_path": str(txt_path.relative_to(out_path)),
            })

        spk_summary = {
            "speaker_id": spk_id,
            "speaker_label": f"Speaker_{spk_id:02d}",
            "folder": str(spk_dir.relative_to(out_path)),
            "total_lines": len(spk_indices),
            "total_speaking_time_sec": round(total_speaking_time, 2),
            "samples": samples_meta,
            "primary_sample": samples_meta[0] if samples_meta else None,
        }

        (spk_dir / "summary.json").write_text(json.dumps(spk_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["speakers"][f"Speaker_{spk_id:02d}"] = spk_summary

        first_txt = samples_meta[0]["text"] if samples_meta else ""
        print(f"  📁 Speaker_{spk_id:02d} ({len(spk_indices)} lines, {total_speaking_time:.1f}s): {len(samples_meta)} samples saved -> '{first_txt}'")

    # Save manifest
    manifest_file = out_path / "review_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate interactive HTML audio review player
    generate_review_html(manifest, out_path / "review_player.html")

    print(f"\n✅ Extraction complete!")
    print(f"📄 Manifest:    {manifest_file.resolve()}")
    print(f"🌐 Audio Player: {out_path / 'review_player.html'}")
    return manifest


def generate_review_html(manifest: dict, html_path: Path):
    """Creates a standalone HTML page with audio players to listen, confirm, or deny each actor."""
    speakers = manifest.get("speakers", {})

    speaker_cards_html = []
    for spk_key, spk_data in speakers.items():
        samples_html = []
        for s in spk_data.get("samples", []):
            samples_html.append(f"""
            <div class="sample-item">
              <div class="sample-header">
                <span class="sample-rank">#{s['rank']} ({s['duration']:.1f}s)</span>
                <span class="sample-time">[{s['start']:.1f}s - {s['end']:.1f}s]</span>
              </div>
              <div class="sample-text">"{s['text']}"</div>
              <audio controls preload="none" src="{s['wav_path']}"></audio>
            </div>
            """)

        speaker_cards_html.append(f"""
        <div class="speaker-card" id="card-{spk_key}">
          <div class="card-header">
            <div>
              <h3>{spk_key}</h3>
              <p class="stats">{spk_data['total_lines']} lines | {spk_data['total_speaking_time_sec']}s dialogue</p>
            </div>
            <div class="action-buttons">
              <input type="text" placeholder="Actor Name (e.g. Feng Lin)" id="name-{spk_key}" class="actor-input">
              <select id="action-{spk_key}" class="action-select">
                <option value="confirm">Confirm as Actor</option>
                <option value="deny">Deny / Discard (Noise)</option>
                <option value="merge">Merge into other Speaker</option>
              </select>
            </div>
          </div>
          <div class="samples-container">
            {''.join(samples_html)}
          </div>
        </div>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Actor Voice Reference Review Player</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #121316;
      color: #e1e4e8;
      padding: 24px;
      margin: 0;
    }}
    .header {{
      max-width: 1100px;
      margin: 0 auto 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid #30363d;
    }}
    .header h1 {{ margin: 0 0 8px 0; font-size: 26px; color: #58a6ff; }}
    .header p {{ margin: 0; color: #8b949e; font-size: 14px; }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}
    .speaker-card {{
      background: #1c1e24;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 18px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      border-bottom: 1px solid #272a30;
      padding-bottom: 12px;
    }}
    .card-header h3 {{ margin: 0 0 4px 0; font-size: 20px; color: #79c0ff; }}
    .stats {{ margin: 0; font-size: 13px; color: #8b949e; }}
    .action-buttons {{ display: flex; gap: 10px; align-items: center; }}
    .actor-input {{
      background: #0d1117;
      border: 1px solid #30363d;
      color: #fff;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
      width: 200px;
    }}
    .action-select {{
      background: #21262d;
      border: 1px solid #30363d;
      color: #c9d1d9;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
    }}
    .samples-container {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 14px;
    }}
    .sample-item {{
      background: #23272e;
      border: 1px solid #363b44;
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .sample-header {{
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: #8b949e;
    }}
    .sample-rank {{ font-weight: bold; color: #f0883e; }}
    .sample-text {{
      font-size: 14px;
      line-height: 1.4;
      color: #f0f6fc;
      background: #191b20;
      padding: 8px 10px;
      border-radius: 6px;
      min-height: 40px;
    }}
    audio {{
      width: 100%;
      height: 36px;
      outline: none;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>🎙️ Actor Voice Reference Review Player</h1>
    <p>Total Discovered Clusters: {len(speakers)} | Total Dialogue Lines: {manifest.get('total_lines', 0)}</p>
    <p>Listen to each speaker's candidate samples below. You can confirm each actor's identity, rename them, or mark unwanted noise clusters.</p>
  </div>
  <div class="container">
    {''.join(speaker_cards_html)}
  </div>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")


def run_reference_puller(
    vocals_path: str,
    srt_path: str,
    output_dir: str = "output/speaker_candidates",
    num_speakers: Optional[int] = None,
    samples_per_speaker: int = 5,
) -> Dict[str, any]:
    """Complete automated extraction pipeline."""
    entries = parse_srt(srt_path)
    embeddings, valid_indices, audio_mono, sr = extract_vocal_embeddings(vocals_path, entries)
    speaker_map, k = cluster_vocal_profiles(embeddings, valid_indices, entries, num_speakers=num_speakers)
    manifest = save_actor_speech_samples(
        audio_mono=audio_mono,
        sr=sr,
        entries=entries,
        speaker_map=speaker_map,
        num_speakers=k,
        output_dir=output_dir,
        samples_per_speaker=samples_per_speaker,
    )
    return manifest
