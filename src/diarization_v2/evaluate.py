#!/usr/bin/env python3
"""
evaluate.py - Quantitative Evaluation Harness for Diarization v2.

Evaluates predicted actor attribution against human ground-truth labels across
benchmark windows (Window 1: 0:26-1:12; Window 2: 7:19-8:10).
"""

import sys
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict


def evaluate(pred_json_path: str, gt_csv_path: str) -> Dict[str, Any]:
    """
    Computes rigorous metrics:
    - Overall accuracy (excluding OVERLAP)
    - Window 1 (Auditions) & Window 2 (Dorm conversation) accuracy
    - Distinct predicted speakers vs truth count
    - Phantom actors (predicted actors that never spoke in that window)
    - Confusion matrix
    """
    pred_path = Path(pred_json_path)
    gt_path = Path(gt_csv_path)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    # Load predictions
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    
    pred_segments = pred_data.get("segments", pred_data)
    pred_by_orig_id = {s["orig_id"]: s.get("actor_id", "Unknown") for s in pred_segments if "orig_id" in s}

    # Load GT
    gt_rows = []
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_rows.append(r)

    total_evaluable = 0
    total_correct = 0

    per_window = defaultdict(lambda: {"total": 0, "correct": 0, "gt_actors": set(), "pred_actors": set()})
    confusion = defaultdict(Counter)

    for r in gt_rows:
        orig_id = int(r["orig_id"])
        win = r.get("window", "default")
        true_act = r.get("true_actor", "").strip()

        # If GT is not yet labeled, skip
        if not true_act:
            continue

        pred_act = pred_by_orig_id.get(orig_id, "UNASSIGNED")

        if true_act == "OVERLAP":
            continue

        total_evaluable += 1
        is_corr = (pred_act == true_act)
        if is_corr:
            total_correct += 1

        per_window[win]["total"] += 1
        if is_corr:
            per_window[win]["correct"] += 1

        per_window[win]["gt_actors"].add(true_act)
        per_window[win]["pred_actors"].add(pred_act)

        confusion[true_act][pred_act] += 1

    overall_acc = (total_correct / total_evaluable * 100.0) if total_evaluable > 0 else 0.0

    windows_summary = {}
    for win, d in per_window.items():
        w_acc = (d["correct"] / d["total"] * 100.0) if d["total"] > 0 else 0.0
        phantom = list(d["pred_actors"] - d["gt_actors"])
        windows_summary[win] = {
            "total_lines": d["total"],
            "correct_lines": d["correct"],
            "accuracy_pct": round(w_acc, 2),
            "distinct_gt_actors": len(d["gt_actors"]),
            "distinct_pred_actors": len(d["pred_actors"]),
            "phantom_actors": phantom,
            "phantom_count": len(phantom)
        }

    return {
        "pred_file": str(pred_path),
        "gt_file": str(gt_path),
        "total_evaluable_lines": total_evaluable,
        "total_correct": total_correct,
        "overall_accuracy_pct": round(overall_acc, 2),
        "windows": windows_summary,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()}
    }


def main():
    if len(sys.argv) < 3:
        pred_file = "output/ep1_lower_30m_diarized.json"
        gt_file = "output/diar_v2/ground_truth.csv"
        if not Path(gt_file).exists():
            gt_file = "output/diar_v2/ground_truth_template.csv"
    else:
        pred_file = sys.argv[1]
        gt_file = sys.argv[2]

    print("=" * 70)
    print("📊 DIARIZATION V2 EVALUATION HARNESS")
    print(f"Prediction:   {pred_file}")
    print(f"Ground Truth: {gt_file}")
    print("=" * 70)

    try:
        results = evaluate(pred_file, gt_file)
        print(f"\nOverall Evaluated Lines : {results['total_evaluable_lines']}")
        if results['total_evaluable_lines'] == 0:
            print("⚠️ No labeled ground truth rows found in CSV! (All true_actor fields are empty).")
            print("👉 Open output/diar_v2/label_tool.html to label the benchmark lines.")
            return

        print(f"Overall Accuracy        : {results['overall_accuracy_pct']}% ({results['total_correct']}/{results['total_evaluable_lines']})")
        print("\nWindow Breakdown:")
        for w, d in results["windows"].items():
            print(f"  [{w}]")
            print(f"    Accuracy         : {d['accuracy_pct']}% ({d['correct_lines']}/{d['total_lines']})")
            print(f"    Speakers (GT/Pred): {d['distinct_gt_actors']} GT vs {d['distinct_pred_actors']} Predicted")
            print(f"    Phantom Actors   : {d['phantom_count']} ({', '.join(d['phantom_actors'])})")

        out_eval = Path("output/diar_v2/eval_baseline.json")
        out_eval.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n💾 Saved evaluation report to {out_eval}")
    except Exception as ex:
        print(f"Error during evaluation: {ex}")


if __name__ == "__main__":
    main()
