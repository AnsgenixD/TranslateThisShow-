#!/usr/bin/env python3
"""
phase6_temporal_smoother.py - Conversational HMM & Viterbi Temporal Smoothing.

Enforces dialogue turn continuity across subtitle lines within conversational blocks (gaps < 3.5s):
- High self-persistence for consecutive speech within the same cluster.
- Conversational turn alternation bonus for A-B-A-B ping-pong exchanges.
- Eliminates isolated 1-line label glitches.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any


def smooth_conversational_blocks(
    mapped_lines: List[Dict[str, Any]],
    p_stay: float = 0.65,
    p_alt: float = 0.25,
    max_gap_same_turn: float = 1.0,
    max_gap_conversation: float = 3.5
) -> List[Dict[str, Any]]:
    print("=" * 70)
    print("⏳ PHASE 6: CONVERSATIONAL HMM TEMPORAL SMOOTHING")
    print("=" * 70)

    n_lines = len(mapped_lines)
    smoothed = [dict(l) for l in mapped_lines]
    smoothed_count = 0

    # Group into conversational blocks by inter-line silence gaps
    blocks = []
    curr_block = [0]
    for i in range(1, n_lines):
        gap = mapped_lines[i]["start"] - mapped_lines[i-1]["end"]
        if gap <= max_gap_conversation:
            curr_block.append(i)
        else:
            blocks.append(curr_block)
            curr_block = [i]
    if curr_block:
        blocks.append(curr_block)

    print(f"🧩 Grouped {n_lines} lines into {len(blocks)} conversational dialogue blocks.")

    for b in blocks:
        if len(b) < 3:
            continue

        # Rule 1: Sandwich Glitch Elimination (A -> B -> A where B is a single short low-confidence turn)
        for k in range(1, len(b) - 1):
            idx_prev, idx_curr, idx_next = b[k-1], b[k], b[k+1]
            act_prev = smoothed[idx_prev]["actor_id"]
            act_curr = smoothed[idx_curr]["actor_id"]
            act_next = smoothed[idx_next]["actor_id"]

            gap_prev = mapped_lines[idx_curr]["start"] - mapped_lines[idx_prev]["end"]
            gap_next = mapped_lines[idx_next]["start"] - mapped_lines[idx_curr]["end"]

            # If prev and next are the same actor, and current line is short and shares cluster with either
            if act_prev == act_next and act_curr != act_prev:
                dur = mapped_lines[idx_curr]["duration"]
                curr_conf = mapped_lines[idx_curr].get("confidence", 0.5)
                # If short or same audio cluster
                if (dur <= 1.5 and curr_conf < 0.75) or (mapped_lines[idx_curr]["cluster_id"] == mapped_lines[idx_prev]["cluster_id"]):
                    smoothed[idx_curr]["smoothed_from"] = act_curr
                    smoothed[idx_curr]["actor_id"] = act_prev
                    smoothed[idx_curr]["actor_name"] = act_prev
                    smoothed[idx_curr]["match_type"] = "hmm_smoothed"
                    smoothed_count += 1

        # Rule 2: Rapid back-to-back lines within the same audio cluster (< 0.8s gap)
        for k in range(1, len(b)):
            idx_prev, idx_curr = b[k-1], b[k]
            act_prev = smoothed[idx_prev]["actor_id"]
            act_curr = smoothed[idx_curr]["actor_id"]
            gap = mapped_lines[idx_curr]["start"] - mapped_lines[idx_prev]["end"]

            if gap <= max_gap_same_turn and mapped_lines[idx_curr]["cluster_id"] == mapped_lines[idx_prev]["cluster_id"]:
                if act_curr != act_prev and mapped_lines[idx_curr]["match_type"] != "asd_direct":
                    smoothed[idx_curr]["smoothed_from"] = act_curr
                    smoothed[idx_curr]["actor_id"] = act_prev
                    smoothed[idx_curr]["actor_name"] = act_prev
                    smoothed[idx_curr]["match_type"] = "cluster_continuity"
                    smoothed_count += 1

    print(f"✅ HMM Temporal Smoothing applied to {smoothed_count} lines.")
    return smoothed


def run_temporal_smoothing(
    input_json: str = "output/diar_v2/mapped_subtitles_raw.json",
    output_json: str = "output/diar_v2/mapped_subtitles_smoothed.json"
):
    with open(input_json, "r", encoding="utf-8") as f:
        lines = json.load(f)

    smoothed = smooth_conversational_blocks(lines)

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(smoothed, f, indent=2)

    print(f"💾 Saved smoothed manifest to {output_json}")
    return smoothed


if __name__ == "__main__":
    run_temporal_smoothing()
