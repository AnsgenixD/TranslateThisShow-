#!/usr/bin/env python3
"""
generate_gt_template_and_tool.py
Extracts official subtitle lines for Window 1 (26-72s) and Window 2 (439-491s),
creates ground_truth_template.csv, and builds an interactive HTML labeling tool.
"""

import os
import re
import csv
import json
from pathlib import Path

def parse_srt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(
        r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\d+\n|\Z)"
    )
    entries = []
    for m in pattern.finditer(text):
        idx, s_str, e_str, content = m.groups()
        h, mn, s_ms = s_str.split(":")
        s, ms = s_ms.split(",")
        start_sec = int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        h, mn, s_ms = e_str.split(":")
        s, ms = s_ms.split(",")
        end_sec = int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        entries.append({
            "orig_id": int(idx),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration": round(end_sec - start_sec, 3),
            "text": content.strip().replace("\n", " ")
        })
    return entries

cn_files = list(Path("subtitles/cn").glob("*EP1-下*"))
en_files = list(Path("subtitles/en").glob("*EP1-下*"))

if not cn_files or not en_files:
    raise FileNotFoundError("Could not find official subtitle files for EP1-下")

cn_entries = {e["orig_id"]: e for e in parse_srt(str(cn_files[0])) if e["start"] <= 1800.0}
en_entries = {e["orig_id"]: e for e in parse_srt(str(en_files[0])) if e["start"] <= 1800.0}

common_ids = sorted(set(cn_entries.keys()) & set(en_entries.keys()))

selected_lines = []
for cid in common_ids:
    c = cn_entries[cid]
    e = en_entries[cid]
    s = c["start"]
    # Window 1: 26s - 72s; Window 2: 439s - 491s
    in_w1 = 26.0 <= s <= 72.0
    in_w2 = 439.0 <= s <= 491.0
    if in_w1 or in_w2:
        window_name = "W1_Intro_Auditions" if in_w1 else "W2_Dorm_Conversation"
        selected_lines.append({
            "orig_id": cid,
            "window": window_name,
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "zh": c["text"],
            "en": e["text"],
            "true_actor": ""
        })

print(f"Extracted {len(selected_lines)} lines across Window 1 and Window 2.")

out_csv = Path("output/diar_v2/ground_truth_template.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["orig_id", "window", "start", "end", "duration", "zh", "en", "true_actor"])
    writer.writeheader()
    writer.writerows(selected_lines)

print(f"Saved template CSV to {out_csv}")

# Actor options
actors = [f"Person_{i:02d}" for i in range(1, 13)]
special_actors = ["OFFSCREEN_UNKNOWN", "NARRATOR", "OVERLAP"]
all_actor_options = actors + special_actors

# Load actor genders & avatars if available
roster_path = Path("output/face_roster/confirmed_actor_roster.json")
roster_data = json.loads(roster_path.read_text(encoding="utf-8")).get("confirmed_actors", {}) if roster_path.exists() else {}

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Diarization v2 Ground Truth Annotation Tool</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f1117; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  header {{ background: #1a1f2c; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2d3748; }}
  header h1 {{ font-size: 18px; color: #60a5fa; font-weight: 600; }}
  .btn {{ background: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px; transition: background 0.2s; }}
  .btn:hover {{ background: #2563eb; }}
  .btn-export {{ background: #10b981; }}
  .btn-export:hover {{ background: #059669; }}
  .main {{ display: flex; flex: 1; overflow: hidden; }}
  .video-panel {{ width: 45%; padding: 16px; display: flex; flex-direction: column; background: #141824; border-right: 1px solid #2d3748; }}
  video {{ width: 100%; max-height: 400px; background: #000; border-radius: 8px; border: 1px solid #334155; }}
  .actor-roster {{ margin-top: 16px; flex: 1; overflow-y: auto; background: #1e2433; border-radius: 8px; padding: 12px; border: 1px solid #2d3748; }}
  .actor-roster h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 10px; }}
  .actor-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
  .actor-card {{ display: flex; flex-direction: column; align-items: center; background: #0f1117; padding: 8px; border-radius: 6px; border: 1px solid #334155; font-size: 11px; text-align: center; }}
  .actor-card img {{ width: 50px; height: 50px; border-radius: 50%; object-fit: cover; margin-bottom: 4px; border: 2px solid #3b82f6; }}
  .list-panel {{ width: 55%; display: flex; flex-direction: column; overflow: hidden; }}
  .table-header {{ display: grid; grid-template-columns: 50px 80px 1fr 140px; padding: 10px 16px; background: #1a1f2c; font-size: 12px; font-weight: 600; color: #94a3b8; border-bottom: 1px solid #2d3748; }}
  .lines-list {{ flex: 1; overflow-y: auto; }}
  .line-row {{ display: grid; grid-template-columns: 50px 80px 1fr 140px; padding: 10px 16px; border-bottom: 1px solid #1e293b; align-items: center; font-size: 13px; cursor: pointer; transition: background 0.15s; }}
  .line-row:hover {{ background: #1e293b; }}
  .line-row.active {{ background: #1e3a8a; border-left: 4px solid #60a5fa; }}
  .line-row.labeled {{ border-left: 4px solid #10b981; }}
  .time-badge {{ font-family: monospace; font-size: 11px; color: #94a3b8; }}
  .text-box {{ padding-right: 12px; }}
  .zh-text {{ font-weight: 500; color: #f8fafc; margin-bottom: 2px; }}
  .en-text {{ font-size: 12px; color: #94a3b8; }}
  select.actor-select {{ width: 100%; padding: 6px 8px; background: #0f172a; border: 1px solid #475569; border-radius: 4px; color: #e2e8f0; font-size: 12px; }}
  select.actor-select:focus {{ border-color: #3b82f6; outline: none; }}
  .stats-badge {{ font-size: 13px; color: #94a3b8; }}
  .window-header {{ background: #232d42; color: #38bdf8; font-weight: 600; padding: 8px 16px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
</style>
</head>
<body>
  <header>
    <div>
      <h1>Diarization v2 Ground Truth Annotation Studio</h1>
      <span class="stats-badge" id="statsBadge">Loaded {len(selected_lines)} lines</span>
    </div>
    <div style="display: flex; gap: 12px;">
      <button class="btn btn-export" onclick="exportCSV()">Export ground_truth.csv</button>
    </div>
  </header>
  <div class="main">
    <div class="video-panel">
      <video id="videoPlayer" controls src="../ep1_lower_30m.mp4" playsinline></video>
      <div class="actor-roster">
        <h3>12 Confirmed Actor Faces (Click to assign to selected line)</h3>
        <div class="actor-grid">
"""

for act in actors:
    img_src = f"../visual_actors/{act}/avatar.jpg"
    html_content += f"""          <div class="actor-card" onclick="quickAssign('{act}')" style="cursor: pointer;">
            <img src="{img_src}" alt="{act}" onerror="this.style.display='none'">
            <strong>{act}</strong>
          </div>
"""

html_content += """        </div>
      </div>
    </div>
    <div class="list-panel">
      <div class="table-header">
        <div>#</div>
        <div>Time</div>
        <div>Dialogue (ZH / EN)</div>
        <div>Assigned Actor</div>
      </div>
      <div class="lines-list" id="linesList">
"""

current_win = ""
for item in selected_lines:
    if item["window"] != current_win:
        current_win = item["window"]
        html_content += f'        <div class="window-header">{current_win.replace("_", " ")}</div>\n'
    
    oid = item["orig_id"]
    s = item["start"]
    e = item["end"]
    time_str = f"{int(s//60):02d}:{s%60:05.2f}"
    zh = item["zh"]
    en = item["en"]
    
    options_html = '<option value="">-- Select Actor --</option>'
    for opt in all_actor_options:
        options_html += f'<option value="{opt}">{opt}</option>'

    html_content += f"""        <div class="line-row" id="row_{oid}" onclick="seekVideo({s}, {oid})">
          <div>{oid}</div>
          <div class="time-badge">{time_str}</div>
          <div class="text-box">
            <div class="zh-text">{zh}</div>
            <div class="en-text">{en}</div>
          </div>
          <div>
            <select class="actor-select" id="sel_{oid}" onchange="onSelectChange({oid})" onclick="event.stopPropagation()">
              {options_html}
            </select>
          </div>
        </div>
"""

html_content += """      </div>
    </div>
  </div>

<script>
  const lines = """ + json.dumps(selected_lines, ensure_ascii=False) + """;
  const player = document.getElementById("videoPlayer");
  let activeOid = null;

  function seekVideo(timeSec, oid) {
    player.currentTime = Math.max(0, timeSec - 0.1);
    player.play();
    highlightRow(oid);
  }

  function highlightRow(oid) {
    activeOid = oid;
    document.querySelectorAll(".line-row").forEach(r => r.classList.remove("active"));
    const row = document.getElementById("row_" + oid);
    if (row) {
      row.classList.add("active");
      row.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function onSelectChange(oid) {
    const sel = document.getElementById("sel_" + oid);
    const row = document.getElementById("row_" + oid);
    const line = lines.find(l => l.orig_id === oid);
    if (line) {
      line.true_actor = sel.value;
    }
    if (sel.value) {
      row.classList.add("labeled");
    } else {
      row.classList.remove("labeled");
    }
    updateStats();
  }

  function quickAssign(actorId) {
    if (!activeOid) return;
    const sel = document.getElementById("sel_" + activeOid);
    if (sel) {
      sel.value = actorId;
      onSelectChange(activeOid);
      // Auto advance to next line
      const currIdx = lines.findIndex(l => l.orig_id === activeOid);
      if (currIdx !== -1 && currIdx + 1 < lines.length) {
        const nextLine = lines[currIdx + 1];
        seekVideo(nextLine.start, nextLine.orig_id);
      }
    }
  }

  function updateStats() {
    const labeledCount = lines.filter(l => l.true_actor && l.true_actor.length > 0).length;
    document.getElementById("statsBadge").innerText = `Progress: ${labeledCount} / ${lines.length} lines labeled`;
  }

  function exportCSV() {
    let csvContent = "orig_id,window,start,end,duration,zh,en,true_actor\\n";
    for (const l of lines) {
      const row = [
        l.orig_id,
        l.window,
        l.start,
        l.end,
        l.duration,
        `"${(l.zh || '').replace(/"/g, '""')}"`,
        `"${(l.en || '').replace(/"/g, '""')}"`,
        `"${(l.true_actor || '').replace(/"/g, '""')}"`
      ];
      csvContent += row.join(",") + "\\n";
    }
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ground_truth.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
</script>
</body>
</html>
"""

out_html = Path("output/diar_v2/label_tool.html")
out_html.write_text(html_content, encoding="utf-8")
print(f"Saved interactive labeling tool to {out_html}")
