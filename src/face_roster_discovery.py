"""
Stage 1: Face Roster & Screen Time Discovery.

Powered by InsightFace (RetinaFace + ArcFace 512-D) with:
1. Temporal face tracklet extraction (quality-weighted centroid per continuous camera shot).
2. Spatial "Cannot-Link" mutual exclusion constraints (faces in same frame can never merge).
3. 2-Stage Constrained Agglomerative Clustering (Tracklet Complete Linkage + Centroid Aggregation).
4. True cumulative Screen Time calculation across the entire episode.
5. Interactive Web Selector with diverse cross-timeline crops, 1-click merging, role toggling, and JSON export.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute Intersection-over-Union between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(interArea / (boxAArea + boxBArea - interArea + 1e-6))


def extract_context_crop(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Crop face with contextual padding (head, hair, and neckline) for natural UI recognition."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_x = bw * 0.25
    pad_y = bh * 0.35
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return frame[int(y1):int(y2), int(x1):int(x2)]
    return crop


def discover_face_roster(
    video_path: str = "output/ep1_lower_30m.mp4",
    output_dir: str = "output/face_roster",
    sample_interval_sec: float = 1.5,
    min_face_area: int = 1600,
    min_det_confidence: float = 0.60,
    stageA_threshold: float = 0.44,
    stageB_threshold: float = 0.54,
    main_cast_min_screen_time_sec: float = 25.0,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🎬 STAGE 1: INTELLIGENT FACE ROSTER & SCREEN TIME DISCOVERY (ArcFace 512-D)")
    print(f"📹 Video: {video_path}")
    print(f"⏱️ Interval: Every {sample_interval_sec:.1f}s")
    print(f"📂 Output: {out_dir.resolve()}")
    print("=" * 70)

    # Initialize InsightFace with RetinaFace detection and ArcFace 512-D recognition
    print("\n🧠 Loading InsightFace ArcFace models (CPU / CoreML)...")
    app = FaceAnalysis(
        name="buffalo_s",
        root=str(Path.home() / ".insightface"),
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / max(1.0, fps)
    timestamps = np.arange(0, duration_sec, sample_interval_sec)

    print(f"🔍 Scanning {len(timestamps)} frames across {duration_sec / 60:.1f} minutes...")
    t0 = time.time()

    active_tracklets: List[Dict[str, Any]] = []
    completed_tracklets: List[Dict[str, Any]] = []
    total_detections = 0

    for f_idx, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        faces = app.get(frame)
        valid_faces = []
        for f in faces:
            bbox = f.bbox
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area >= min_face_area and f.det_score >= min_det_confidence:
                # Aligned 112x112 canonical crop
                try:
                    aligned_crop = face_align.norm_crop(frame, landmark=f.kps, image_size=112)
                except Exception:
                    aligned_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                
                context_crop = extract_context_crop(frame, bbox)
                valid_faces.append({
                    "bbox": bbox,
                    "score": float(f.det_score),
                    "area": float(area),
                    "normed_emb": f.normed_embedding,
                    "aligned_crop": aligned_crop,
                    "context_crop": context_crop,
                })

        total_detections += len(valid_faces)

        # Match detections to active tracklets (temporal continuity within camera shot)
        matched_tracklets = set()
        matched_faces = set()

        for t_idx, tr in enumerate(active_tracklets):
            best_match = None
            best_score = -1.0
            for f_i, f in enumerate(valid_faces):
                if f_i in matched_faces:
                    continue
                iou = compute_iou(tr["last_bbox"], f["bbox"])
                cos_sim = float(np.dot(tr["emb"], f["normed_emb"]))
                
                # Match criterion: smooth spatial overlap OR moderate overlap with high facial similarity
                if iou > 0.35 or (iou > 0.15 and cos_sim > 0.65):
                    score = iou + cos_sim
                    if score > best_score:
                        best_score = score
                        best_match = f_i

            if best_match is not None:
                f = valid_faces[best_match]
                tr["frames"].append({
                    "ts": float(ts),
                    "bbox": [float(b) for b in f["bbox"]],
                    "score": f["score"],
                    "area": f["area"],
                    "emb": f["normed_emb"],
                    "aligned_crop": f["aligned_crop"],
                    "context_crop": f["context_crop"],
                })
                tr["last_bbox"] = f["bbox"]
                tr["last_ts"] = float(ts)

                # Recompute quality-weighted running centroid
                all_embs = np.array([item["emb"] for item in tr["frames"]])
                all_scores = np.array([item["score"] for item in tr["frames"]])[:, None]
                weighted_mean = np.sum(all_embs * all_scores, axis=0)
                tr["emb"] = weighted_mean / (np.linalg.norm(weighted_mean) + 1e-6)

                matched_tracklets.add(t_idx)
                matched_faces.add(best_match)

        # Close tracklets that received no detection in this frame
        new_active = []
        for t_idx, tr in enumerate(active_tracklets):
            if t_idx in matched_tracklets:
                new_active.append(tr)
            else:
                completed_tracklets.append(tr)
        active_tracklets = new_active

        # Initialize new tracklets for unmatched detections
        for f_i, f in enumerate(valid_faces):
            if f_i not in matched_faces:
                active_tracklets.append({
                    "start_ts": float(ts),
                    "last_ts": float(ts),
                    "last_bbox": f["bbox"],
                    "emb": f["normed_emb"],
                    "frames": [{
                        "ts": float(ts),
                        "bbox": [float(b) for b in f["bbox"]],
                        "score": f["score"],
                        "area": f["area"],
                        "emb": f["normed_emb"],
                        "aligned_crop": f["aligned_crop"],
                        "context_crop": f["context_crop"],
                    }],
                })

        if (f_idx + 1) % 200 == 0 or (f_idx + 1) == len(timestamps):
            elapsed = time.time() - t0
            fps_proc = (f_idx + 1) / max(0.1, elapsed)
            print(f"  -> Scanned {f_idx + 1}/{len(timestamps)} frames ({total_detections} faces, {len(completed_tracklets) + len(active_tracklets)} tracklets, {fps_proc:.1f} fps)")

    completed_tracklets.extend(active_tracklets)
    cap.release()

    scan_time = time.time() - t0
    print(f"\n✅ Scan finished in {scan_time:.1f}s: {total_detections} face instances grouped into {len(completed_tracklets)} tracklets.")

    if not completed_tracklets:
        print("⚠️ No face tracklets found.")
        return {}

    # Build Cannot-Link Mutual Exclusion Matrix
    print("\n🔗 Building Spatial Cannot-Link Constraints (Faces visible in the same frame)...")
    N = len(completed_tracklets)
    cannot_link = np.zeros((N, N), dtype=bool)
    tracklet_timestamps = [set(item["ts"] for item in t["frames"]) for t in completed_tracklets]

    for i in range(N):
        for j in range(i + 1, N):
            if len(tracklet_timestamps[i].intersection(tracklet_timestamps[j])) > 0:
                cannot_link[i, j] = cannot_link[j, i] = True

    cl_count = int(np.sum(cannot_link) // 2)
    print(f"   -> Enforced {cl_count} spatial mutual exclusion constraints.")

    # 2-Stage Hierarchical Clustering
    print("\n🧬 Stage A: High-Confidence Tracklet Complete Linkage...")
    embs = np.array([t["emb"] for t in completed_tracklets])
    dist_mat = 1.0 - np.dot(embs, embs.T)
    # Penalize cannot-link pairs so complete linkage never merges them
    dist_mat[cannot_link] = 999.0
    np.fill_diagonal(dist_mat, 999.0)

    ac_stageA = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=stageA_threshold,
        metric="precomputed",
        linkage="complete",
    )
    stageA_labels = ac_stageA.fit_predict(dist_mat)

    # Group tracklets into intermediate clusters
    intermediate_clusters: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(stageA_labels):
        intermediate_clusters.setdefault(int(lbl), []).append(idx)

    print(f"   -> Stage A consolidated 420 tracklets into {len(intermediate_clusters)} clean identity seeds.")

    # Stage B: Centroid Merge Across Different Days / Scenes with strict Cannot-Link Verification
    print(f"🧬 Stage B: Centroid-Level Agglomerative Merge (Threshold: {stageB_threshold:.2f})...")
    cids = list(intermediate_clusters.keys())
    cluster_members: Dict[int, List[int]] = {cid: list(intermediate_clusters[cid]) for cid in cids}
    cluster_centroids: Dict[int, np.ndarray] = {}

    for cid, members in cluster_members.items():
        c_embs = embs[members]
        mean_e = np.mean(c_embs, axis=0)
        cluster_centroids[cid] = mean_e / np.linalg.norm(mean_e)

    active_cids = set(cids)
    merged_count = 0

    while True:
        best_dist = stageB_threshold
        best_pair = None
        active_list = list(active_cids)

        for i_idx in range(len(active_list)):
            ci = active_list[i_idx]
            ei = cluster_centroids[ci]
            mi = cluster_members[ci]
            for j_idx in range(i_idx + 1, len(active_list)):
                cj = active_list[j_idx]
                ej = cluster_centroids[cj]
                mj = cluster_members[cj]

                # Check cannot-link constraint between all constituent tracklets
                if cannot_link[np.ix_(mi, mj)].any():
                    continue

                d = 1.0 - float(np.dot(ei, ej))
                if d < best_dist:
                    best_dist = d
                    best_pair = (ci, cj)

        if best_pair is None:
            break

        ci, cj = best_pair
        cluster_members[ci].extend(cluster_members[cj])
        # Update centroid
        new_embs = embs[cluster_members[ci]]
        new_mean = np.mean(new_embs, axis=0)
        cluster_centroids[ci] = new_mean / np.linalg.norm(new_mean)

        del cluster_members[cj]
        del cluster_centroids[cj]
        active_cids.remove(cj)
        merged_count += 1

    print(f"   -> Stage B merged {merged_count} cross-scene identity clusters without violating any spatial constraints.")
    print(f"   -> Final unique identities discovered: {len(active_cids)}")

    # Calculate Cumulative Screen Time & Prepare Roster
    final_identities = []
    for cid in active_cids:
        t_indices = cluster_members[cid]
        c_tracklets = [completed_tracklets[ti] for ti in t_indices]

        # Collect all frames
        all_frames = []
        for tr in c_tracklets:
            all_frames.extend(tr["frames"])

        # Screen time = sum of unique timestamps covered * sample_interval_sec
        unique_timestamps = sorted(list(set(item["ts"] for item in all_frames)))
        screen_time_sec = round(len(unique_timestamps) * sample_interval_sec, 1)

        # Identity centroid
        c_embs = embs[t_indices]
        mean_emb = np.mean(c_embs, axis=0)
        norm_centroid = mean_emb / np.linalg.norm(mean_emb)

        final_identities.append({
            "screen_time_sec": screen_time_sec,
            "tracklets_count": len(c_tracklets),
            "detections_count": len(all_frames),
            "first_seen_sec": round(unique_timestamps[0], 1),
            "last_seen_sec": round(unique_timestamps[-1], 1),
            "centroid_emb": norm_centroid,
            "frames": all_frames,
            "unique_ts": unique_timestamps,
        })

    # Sort descending by screen time
    final_identities.sort(key=lambda x: x["screen_time_sec"], reverse=True)

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_path": video_path,
        "sample_interval_sec": sample_interval_sec,
        "total_identities": len(final_identities),
        "clusters": {},
    }

    print("\n🌟 Exporting Identity Dossiers & Cross-Timeline Face Crops...")

    for rank, item in enumerate(final_identities, 1):
        person_key = f"Person_{rank:02d}"
        p_dir = out_dir / person_key
        p_dir.mkdir(parents=True, exist_ok=True)

        frames = item["frames"]
        # Select up to 4 diverse crops:
        # 1. Best frontal crop (highest det_score)
        # 2. Earliest scene crop
        # 3. Middle scene crop
        # 4. Latest scene crop
        sorted_by_score = sorted(frames, key=lambda f: f["score"], reverse=True)
        sorted_by_ts = sorted(frames, key=lambda f: f["ts"])

        diverse_crops = [sorted_by_score[0]] # primary avatar
        ts_candidates = [
            sorted_by_ts[0],
            sorted_by_ts[len(sorted_by_ts) // 2],
            sorted_by_ts[-1],
        ]
        for cand in ts_candidates:
            if cand not in diverse_crops and len(diverse_crops) < 4:
                diverse_crops.append(cand)

        crop_paths = []
        for c_idx, c_frame in enumerate(diverse_crops, 1):
            crop_name = f"crop_{c_idx}.jpg"
            crop_file = p_dir / crop_name
            # Save context crop (natural head & shoulders)
            img_to_save = c_frame["context_crop"]
            if img_to_save is not None and img_to_save.size > 0:
                cv2.imwrite(str(crop_file), img_to_save)
            else:
                cv2.imwrite(str(crop_file), c_frame["aligned_crop"])
            crop_paths.append(str(crop_file.relative_to(out_dir)))

        st_sec = item["screen_time_sec"]
        st_fmt = f"{int(st_sec // 60)}m {int(st_sec % 60):02d}s"
        is_main = st_sec >= main_cast_min_screen_time_sec

        info_data = {
            "rank": rank,
            "label": person_key,
            "screen_time_sec": st_sec,
            "screen_time_formatted": st_fmt,
            "shots_count": item["tracklets_count"],
            "appearances": item["detections_count"],
            "first_seen_sec": item["first_seen_sec"],
            "last_seen_sec": item["last_seen_sec"],
            "is_main_cast_suggested": is_main,
            "crop_images": crop_paths,
            "primary_avatar": crop_paths[0],
            "embedding": item["centroid_emb"].tolist(),
        }

        (p_dir / "info.json").write_text(json.dumps(info_data, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["clusters"][person_key] = info_data

        if rank <= 12 or is_main:
            cast_badge = "⭐ MAIN CAST" if is_main else "⚠️ EXTRA/CREW"
            print(f"  #{rank:02d} {person_key}: {st_fmt} screen time ({item['detections_count']} frames across {item['tracklets_count']} shots) -> {cast_badge}")

    manifest_file = out_dir / "roster_manifest.json"
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Export default baseline confirmed_actor_roster.json
    baseline_confirmed = {
        "saved_at": manifest["created_at"],
        "video_path": video_path,
        "confirmed_actors": {},
        "ignored_crew_count": 0,
    }
    for k, v in manifest["clusters"].items():
        if v["is_main_cast_suggested"]:
            baseline_confirmed["confirmed_actors"][k] = {
                "key": k,
                "name": k,
                "screen_time_sec": v["screen_time_sec"],
                "screen_time_formatted": v["screen_time_formatted"],
                "appearances": v["appearances"],
                "shots_count": v["shots_count"],
                "first_seen_sec": v["first_seen_sec"],
                "last_seen_sec": v["last_seen_sec"],
                "primary_avatar": v["primary_avatar"],
                "crop_images": v["crop_images"],
                "embedding": v["embedding"],
            }
        else:
            baseline_confirmed["ignored_crew_count"] += 1
    (out_dir / "confirmed_actor_roster.json").write_text(
        json.dumps(baseline_confirmed, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Generate the Interactive Web Selector UI with 1-Click Merging
    generate_roster_html(manifest, out_dir / "actor_roster_selector.html")

    print(f"\n🎉 Stage 1 Face Discovery Complete!")
    print(f"📄 Manifest: {manifest_file.resolve()}")
    print(f"🌐 Selector: {out_dir / 'actor_roster_selector.html'}")
    return manifest


def generate_roster_html(manifest: Dict[str, Any], html_path: Path):
    clusters = manifest.get("clusters", {})
    all_keys = list(clusters.keys())

    main_cards = []
    crew_cards = []

    for key, data in clusters.items():
        is_main = data["is_main_cast_suggested"]
        checked_attr = "checked" if is_main else ""
        card_class = "is-main" if is_main else "is-crew"
        tag_color = "#3fb950" if is_main else "#8b949e"
        tag_text = "Main Cast Candidate" if is_main else "Transient / Extra / Crew"

        # Thumbnails strip
        thumbs_html = "".join([
            f'<img class="thumb-crop" src="{p}" alt="crop" onclick="setPrimaryAvatar(\'{key}\', \'{p}\')">'
            for p in data["crop_images"]
        ])

        # Merge options (all other people)
        merge_options = "".join([
            f'<option value="{other_k}">{other_k} ({clusters[other_k]["screen_time_formatted"]})</option>'
            for other_k in all_keys if other_k != key
        ])

        card_html = f"""
        <div class="person-card {card_class}" id="card-{key}" data-person-key="{key}" data-screen-time="{data['screen_time_sec']}">
          <div class="avatar-box">
            <img class="main-avatar" id="avatar-{key}" src="{data['primary_avatar']}" alt="{key}">
            <div class="thumb-strip" id="thumbs-{key}">
              {thumbs_html}
            </div>
            <span class="avatar-tip">Click thumbnail to switch avatar</span>
          </div>

          <div class="details-box">
            <div class="details-top">
              <div>
                <div class="title-row">
                  <h3 class="person-title" id="title-{key}">#{data['rank']:02d} {key}</h3>
                  <span class="type-tag" id="tag-{key}" style="color: {tag_color}; border-color: {tag_color}55;">{tag_text}</span>
                </div>
                <div class="badges-row">
                  <span class="time-badge" id="badge-time-{key}">⏱️ {data['screen_time_formatted']} Screen Time</span>
                  <span class="shots-badge" id="badge-shots-{key}">📊 {data['appearances']} frames ({data['shots_count']} shots)</span>
                  <span class="range-badge">🕒 Seen {int(data['first_seen_sec']//60)}:{int(data['first_seen_sec']%60):02d} – {int(data['last_seen_sec']//60)}:{int(data['last_seen_sec']%60):02d}</span>
                </div>
              </div>

              <div class="select-col">
                <label class="check-label">
                  <input type="checkbox" id="check-{key}" class="actor-checkbox" {checked_attr} onchange="updateSelectionTally()">
                  <span class="check-text">Is Main Actor</span>
                </label>
              </div>
            </div>

            <div class="details-middle">
              <input type="text" placeholder="Actor Name (e.g. Feng Lin, Yi Lijing)" id="name-{key}" class="actor-name-input">
            </div>

            <div class="details-bottom">
              <div class="merge-tool">
                <span class="merge-label">⚡ Merge into:</span>
                <select id="merge-select-{key}" class="merge-select">
                  <option value="">Select target...</option>
                  {merge_options}
                </select>
                <button class="merge-btn" onclick="executeMerge('{key}')">Merge</button>
              </div>

              <div class="action-buttons">
                <button class="toggle-role-btn" onclick="toggleRole('{key}')">↔️ Move to {'Crew' if is_main else 'Main Cast'}</button>
              </div>
            </div>
          </div>
        </div>
        """
        if is_main:
            main_cards.append(card_html)
        else:
            crew_cards.append(card_html)

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Stage 1: Face Roster & Screen Time Discovery (ArcFace 512-D)</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #090d13;
      color: #c9d1d9;
      margin: 0;
      padding: 0 0 100px 0;
    }}
    .sticky-bar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: #161b22f5;
      backdrop-filter: blur(14px);
      border-bottom: 1px solid #30363d;
      padding: 14px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .sticky-title h1 {{
      margin: 0;
      font-size: 20px;
      color: #58a6ff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .sticky-title p {{
      margin: 4px 0 0 0;
      font-size: 13px;
      color: #8b949e;
    }}
    .bar-actions {{
      display: flex;
      gap: 16px;
      align-items: center;
    }}
    .tally-tag {{
      font-size: 14px;
      color: #7ee787;
      font-weight: 600;
      background: #23863622;
      border: 1px solid #23863666;
      padding: 6px 14px;
      border-radius: 20px;
    }}
    .confirm-btn {{
      background: #238636;
      color: #fff;
      border: 1px solid rgba(240,246,252,0.1);
      padding: 9px 20px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(35,134,54,0.4);
      transition: background 0.15s, transform 0.1s;
    }}
    .confirm-btn:hover {{ background: #2ea043; transform: translateY(-1px); }}
    .container {{
      max-width: 1080px;
      margin: 24px auto;
      padding: 0 16px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}
    .section-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #21262d;
      padding-bottom: 10px;
      margin-top: 14px;
    }}
    .section-header h2 {{ margin: 0; font-size: 18px; color: #f0f6fc; }}
    .section-header span {{ font-size: 13px; color: #8b949e; }}
    .cards-list {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .person-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 18px;
      display: flex;
      gap: 20px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.4);
      transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .person-card.is-main {{ border-color: #388bfd55; }}
    .person-card:hover {{ border-color: #58a6ff; box-shadow: 0 6px 20px rgba(88,166,255,0.15); }}
    .person-card.is-merged {{ opacity: 0.4; filter: grayscale(80%); pointer-events: none; display: none; }}
    .avatar-box {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      width: 136px;
      flex-shrink: 0;
    }}
    .main-avatar {{
      width: 124px;
      height: 124px;
      border-radius: 10px;
      object-fit: cover;
      border: 2px solid #58a6ff;
      background: #0d1117;
    }}
    .thumb-strip {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: center;
    }}
    .thumb-crop {{
      width: 32px;
      height: 32px;
      border-radius: 4px;
      object-fit: cover;
      border: 1px solid #30363d;
      cursor: pointer;
      transition: transform 0.1s, border-color 0.1s;
    }}
    .thumb-crop:hover {{
      transform: scale(1.15);
      border-color: #58a6ff;
    }}
    .avatar-tip {{
      font-size: 10px;
      color: #6e7681;
      text-align: center;
    }}
    .details-box {{
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 14px;
    }}
    .details-top {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }}
    .title-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 6px;
    }}
    .person-title {{ margin: 0; font-size: 19px; color: #f0f6fc; font-weight: 600; }}
    .type-tag {{
      border: 1px solid #30363d;
      padding: 3px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 500;
    }}
    .badges-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .time-badge {{
      background: #1f2937;
      color: #7ee787;
      border: 1px solid #23863644;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
    }}
    .shots-badge, .range-badge {{
      background: #21262d;
      color: #8b949e;
      border: 1px solid #30363d;
      padding: 4px 9px;
      border-radius: 12px;
      font-size: 11px;
    }}
    .check-label {{
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      user-select: none;
      background: #21262d;
      border: 1px solid #30363d;
      padding: 8px 14px;
      border-radius: 8px;
      transition: background 0.15s;
    }}
    .check-label:hover {{ background: #282e37; }}
    .actor-checkbox {{
      width: 18px;
      height: 18px;
      cursor: pointer;
      accent-color: #238636;
    }}
    .check-text {{ font-weight: 600; font-size: 13px; color: #f0f6fc; }}
    .details-middle {{
      display: flex;
      gap: 12px;
    }}
    .actor-name-input {{
      background: #0d1117;
      border: 1px solid #30363d;
      color: #fff;
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      width: 320px;
    }}
    .actor-name-input:focus {{
      outline: none;
      border-color: #58a6ff;
    }}
    .details-bottom {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid #21262d;
      padding-top: 10px;
    }}
    .merge-tool {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .merge-label {{ font-size: 12px; color: #8b949e; font-weight: 500; }}
    .merge-select {{
      background: #0d1117;
      border: 1px solid #30363d;
      color: #c9d1d9;
      padding: 5px 10px;
      border-radius: 6px;
      font-size: 12px;
      max-width: 220px;
    }}
    .merge-btn {{
      background: #388bfd22;
      color: #58a6ff;
      border: 1px solid #388bfd55;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    .merge-btn:hover {{ background: #388bfd44; }}
    .toggle-role-btn {{
      background: #21262d;
      color: #8b949e;
      border: 1px solid #30363d;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
    }}
    .toggle-role-btn:hover {{ color: #c9d1d9; border-color: #8b949e; }}
    .crew-details {{
      background: #161b2280;
      border: 1px dashed #30363d;
      border-radius: 8px;
      padding: 14px 20px;
      margin-top: 14px;
      cursor: pointer;
    }}
    .crew-details summary {{
      font-size: 14px;
      color: #8b949e;
      outline: none;
      font-weight: 500;
    }}
  </style>
</head>
<body>
  <div class="sticky-bar">
    <div class="sticky-title">
      <h1>👥 Stage 1: Face Roster & Screen Time Discovery</h1>
      <p>Confirm the real actors below. Use "Merge" to combine any duplicate angles/days. Unchecked people (crew/extras) will be discarded.</p>
    </div>
    <div class="bar-actions">
      <span class="tally-tag" id="tally-tag">Loading tally...</span>
      <button class="confirm-btn" onclick="saveConfirmedRoster()">🚀 Confirm Roster & Save</button>
    </div>
  </div>

  <div class="container">
    <div class="section-header">
      <h2>⭐ Main Cast Candidates (Ranked by Screen Time)</h2>
      <span id="main-count-tag">{len(main_cards)} recurring characters detected (&ge; 25s)</span>
    </div>
    <div class="cards-list" id="main-cards-container">
      {''.join(main_cards)}
    </div>

    <details class="crew-details">
      <summary id="crew-summary-tag">⚠️ Backstage Crew & Transient Extras ({len(crew_cards)} faces with &lt; 25s screen time) — Click to expand</summary>
      <div class="cards-list" id="crew-cards-container" style="margin-top: 16px;">
        {''.join(crew_cards)}
      </div>
    </details>
  </div>

  <script>
    let ROSTER_DATA = {json.dumps(clusters, ensure_ascii=False)};

    function updateSelectionTally() {{
      const checkedBoxes = document.querySelectorAll(".actor-checkbox:checked");
      const tallyEl = document.getElementById("tally-tag");
      tallyEl.textContent = `${{checkedBoxes.length}} Actors Selected`;
    }}

    function setPrimaryAvatar(personKey, newSrc) {{
      const avatarImg = document.getElementById(`avatar-${{personKey}}`);
      if (avatarImg) {{
        avatarImg.src = newSrc;
        ROSTER_DATA[personKey].primary_avatar = newSrc;
      }}
    }}

    function toggleRole(personKey) {{
      const card = document.getElementById(`card-${{personKey}}`);
      const mainContainer = document.getElementById("main-cards-container");
      const crewContainer = document.getElementById("crew-cards-container");
      const checkbox = document.getElementById(`check-${{personKey}}`);
      const tag = document.getElementById(`tag-${{personKey}}`);
      const toggleBtn = card.querySelector(".toggle-role-btn");

      const isMainCurrently = card.classList.contains("is-main");
      if (isMainCurrently) {{
        card.classList.remove("is-main");
        card.classList.add("is-crew");
        crewContainer.appendChild(card);
        checkbox.checked = false;
        tag.textContent = "Transient / Extra / Crew";
        tag.style.color = "#8b949e";
        tag.style.borderColor = "#8b949e55";
        toggleBtn.textContent = "↔️ Move to Main Cast";
      }} else {{
        card.classList.remove("is-crew");
        card.classList.add("is-main");
        mainContainer.appendChild(card);
        checkbox.checked = true;
        tag.textContent = "Main Cast Candidate";
        tag.style.color = "#3fb950";
        tag.style.borderColor = "#3fb95055";
        toggleBtn.textContent = "↔️ Move to Crew";
      }}
      updateSelectionTally();
    }}

    function executeMerge(sourceKey) {{
      const select = document.getElementById(`merge-select-${{sourceKey}}`);
      const targetKey = select.value;
      if (!targetKey) {{
        alert("Please select a target person to merge into.");
        return;
      }}

      if (!confirm(`Are you sure you want to merge ${{sourceKey}} into ${{targetKey}}?\\nAll screen time, shots, and face crops will be combined.`)) {{
        return;
      }}

      const sourceData = ROSTER_DATA[sourceKey];
      const targetData = ROSTER_DATA[targetKey];

      // Combine stats
      targetData.screen_time_sec = +(targetData.screen_time_sec + sourceData.screen_time_sec).toFixed(1);
      targetData.appearances += sourceData.appearances;
      targetData.shots_count += sourceData.shots_count;
      targetData.first_seen_sec = Math.min(targetData.first_seen_sec, sourceData.first_seen_sec);
      targetData.last_seen_sec = Math.max(targetData.last_seen_sec, sourceData.last_seen_sec);

      // Format screen time
      const m = Math.floor(targetData.screen_time_sec / 60);
      const s = Math.floor(targetData.screen_time_sec % 60);
      targetData.screen_time_formatted = `${{m}}m ${{s.toString().padStart(2, '0')}}s`;

      // Combine embeddings (average)
      if (sourceData.embedding && targetData.embedding) {{
        for (let i = 0; i < targetData.embedding.length; i++) {{
          targetData.embedding[i] = (targetData.embedding[i] + sourceData.embedding[i]) / 2.0;
        }}
      }}

      // Combine crops
      const targetThumbs = document.getElementById(`thumbs-${{targetKey}}`);
      sourceData.crop_images.forEach(cropPath => {{
        if (!targetData.crop_images.includes(cropPath)) {{
          targetData.crop_images.push(cropPath);
          const img = document.createElement("img");
          img.className = "thumb-crop";
          img.src = cropPath;
          img.alt = "crop";
          img.onclick = () => setPrimaryAvatar(targetKey, cropPath);
          targetThumbs.appendChild(img);
        }}
      }});

      // Update Target Card UI
      const badgeTime = document.getElementById(`badge-time-${{targetKey}}`);
      if (badgeTime) badgeTime.textContent = `⏱️ ${{targetData.screen_time_formatted}} Screen Time`;

      const badgeShots = document.getElementById(`badge-shots-${{targetKey}}`);
      if (badgeShots) badgeShots.textContent = `📊 ${{targetData.appearances}} frames (${{targetData.shots_count}} shots)`;

      // Hide Source Card
      const sourceCard = document.getElementById(`card-${{sourceKey}}`);
      sourceCard.classList.add("is-merged");
      const sourceCheckbox = document.getElementById(`check-${{sourceKey}}`);
      if (sourceCheckbox) sourceCheckbox.checked = false;

      // Mark in data
      sourceData.is_merged_into = targetKey;

      updateSelectionTally();
      alert(`✅ Successfully merged ${{sourceKey}} into ${{targetKey}}! ${{targetKey}} now has ${{targetData.screen_time_formatted}} screen time.`);
    }}

    function saveConfirmedRoster() {{
      const selected = {{
        saved_at: new Date().toISOString(),
        video_path: "{manifest.get('video_path', '')}",
        confirmed_actors: {{}},
        ignored_crew_count: 0
      }};

      Object.keys(ROSTER_DATA).forEach(key => {{
        const item = ROSTER_DATA[key];
        if (item.is_merged_into) return; // Skip merged duplicate

        const isChecked = document.getElementById(`check-${{key}}`).checked;
        const customName = document.getElementById(`name-${{key}}`).value.trim();

        if (isChecked) {{
          selected.confirmed_actors[key] = {{
            key: key,
            name: customName || key,
            screen_time_sec: item.screen_time_sec,
            screen_time_formatted: item.screen_time_formatted,
            appearances: item.appearances,
            shots_count: item.shots_count,
            first_seen_sec: item.first_seen_sec,
            last_seen_sec: item.last_seen_sec,
            primary_avatar: item.primary_avatar,
            crop_images: item.crop_images,
            embedding: item.embedding
          }};
        }} else {{
          selected.ignored_crew_count++;
        }}
      }});

      const blob = new Blob([JSON.stringify(selected, null, 2)], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "confirmed_actor_roster.json";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      alert(`✅ Confirmed ${{Object.keys(selected.confirmed_actors).length}} actors! Downloaded confirmed_actor_roster.json.`);
    }}

    document.addEventListener("DOMContentLoaded", updateSelectionTally);
  </script>
</body>
</html>
"""
    html_path.write_text(html_code, encoding="utf-8")
