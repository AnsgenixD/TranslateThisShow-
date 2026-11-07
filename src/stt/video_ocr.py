"""
video_ocr.py - Video Frame On-Screen Chinese Subtitle & Dialogue Extractor.

Extracts ground-truth broadcast Chinese dialogue directly from the video stream
using lower-third subtitle region change detection and Apple Silicon MLX Qwen3-VL.
Eliminates reliance on editorial SRT human-condensation and subtitle drop artifacts.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class VideoSubtitleOCRExtractor:
    """
    Extracts on-screen Chinese subtitles from video frames via frame-differencing
    transition detection and Qwen3-VL (mlx-vlm).
    """

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        crop_y_ratio: Tuple[float, float] = (0.75, 0.96),
        crop_x_ratio: Tuple[float, float] = (0.05, 0.95),
    ):
        self.model_path = model_path
        self.crop_y_ratio = crop_y_ratio
        self.crop_x_ratio = crop_x_ratio
        self.model = None
        self.processor = None
        self._load_failed = False

    def _ensure_vlm_loaded(self) -> bool:
        if self.model is None and not self._load_failed:
            try:
                from mlx_vlm import load
                print(f"👁️ Loading Vision-Language Model: {self.model_path}...")
                t0 = time.time()
                self.model, self.processor = load(self.model_path)
                print(f"✅ Vision model ready in {time.time() - t0:.2f}s.")
                return True
            except ImportError:
                print("\n💡 [Optional Feature] Video On-Screen Subtitle OCR requires 'mlx-vlm'.")
                print("   To enable direct visual dialogue extraction from video frames:")
                print("     pip install mlx-vlm")
                print("   Recommended vision models:")
                print("     - mlx-community/Qwen3-VL-4B-Instruct-4bit (Fast, lightweight)")
                print("     - mlx-community/Qwen2.5-VL-7B-Instruct-4bit (Higher accuracy)")
                print("   Skipping video visual OCR for now...\n")
                self._load_failed = True
                return False
            except Exception as e:
                print(f"\n💡 [Notice] Vision model '{self.model_path}' not available ({e}).")
                print("   Skipping video visual OCR for now...\n")
                self._load_failed = True
                return False
        return self.model is not None

    def unload(self) -> None:
        """Frees VLM from unified memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            import mlx.core as mx
            mx.clear_cache()

    def detect_subtitle_intervals(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_step_sec: float = 0.20,
        diff_threshold: float = 18.0,
    ) -> List[Dict[str, Any]]:
        """
        Scans video frames across [start_time, end_time] and groups consecutive
        stable frames into subtitle display intervals.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        y0 = int(height * self.crop_y_ratio[0])
        y1 = int(height * self.crop_y_ratio[1])
        x0 = int(width * self.crop_x_ratio[0])
        x1 = int(width * self.crop_x_ratio[1])

        step_frames = max(1, int(round(sample_step_sec * fps)))
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        prev_gray = None
        raw_samples = []

        for f_idx in range(start_frame, end_frame, step_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            sec = f_idx / fps
            crop = frame[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # Check if this frame has text-like high frequency contrast
            std_dev = np.std(gray)
            diff = 0.0
            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, prev_gray)))

            raw_samples.append({
                "sec": round(sec, 3),
                "frame_idx": f_idx,
                "diff": diff,
                "std": std_dev,
                "crop": crop,
            })
            prev_gray = gray

        cap.release()

        # Group samples into continuous subtitle display blocks
        intervals: List[Dict[str, Any]] = []
        cur_block: Optional[Dict[str, Any]] = None

        for s in raw_samples:
            is_new_sub = (s["diff"] > diff_threshold)
            has_contrast = (s["std"] > 16.0)  # low contrast = black/solid bar without text

            if not has_contrast:
                if cur_block:
                    cur_block["end"] = s["sec"]
                    intervals.append(cur_block)
                    cur_block = None
                continue

            if cur_block is None:
                cur_block = {
                    "start": s["sec"],
                    "end": s["sec"] + sample_step_sec,
                    "representative_crop": s["crop"],
                    "rep_sec": s["sec"],
                }
            elif is_new_sub:
                # Subtitle changed
                cur_block["end"] = s["sec"]
                intervals.append(cur_block)
                cur_block = {
                    "start": s["sec"],
                    "end": s["sec"] + sample_step_sec,
                    "representative_crop": s["crop"],
                    "rep_sec": s["sec"],
                }
            else:
                # Same subtitle continuing
                cur_block["end"] = round(s["sec"] + sample_step_sec, 3)

        if cur_block:
            intervals.append(cur_block)

        # Filter out intervals shorter than 0.3s
        valid_intervals = [
            itv for itv in intervals
            if (itv["end"] - itv["start"]) >= 0.35
        ]
        return valid_intervals

    def extract_subtitles_from_video(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float = 120.0,
        temp_dir: str = "output/ocr_cache",
    ) -> List[Dict[str, Any]]:
        """
        Extracts all on-screen Chinese subtitle text within [start_time, end_time].
        Returns a list of structured cues with start, end, text, and visual confidence.
        """
        if not self._ensure_vlm_loaded():
            return []

        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        t_dir = Path(temp_dir)
        t_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎬 Scanning on-screen subtitles in video from {start_time:.1f}s to {end_time:.1f}s...")
        intervals = self.detect_subtitle_intervals(video_path, start_time, end_time)
        print(f"👁️ Detected {len(intervals)} visual subtitle candidates.")

        if not intervals:
            return []
        prompt = (
            "请准确提取并转写截图中显示的底部中文字幕台词文本。\n"
            "规则：只输出台词文本本身，去掉人名冒号前缀，不要任何解释。若无台词或无中文字幕则输出NONE。"
        )
        formatted_prompt = apply_chat_template(
            self.processor,
            config=self.model.config,
            prompt=prompt,
            num_images=1
        )

        extracted_cues: List[Dict[str, Any]] = []

        for idx, itv in enumerate(intervals):
            # Save representative crop to disk
            crop_path = t_dir / f"crop_{idx}_{int(itv['rep_sec']*1000)}.jpg"
            cv2.imwrite(str(crop_path), itv["representative_crop"])

            try:
                res = generate(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    image=str(crop_path),
                    max_tokens=32,
                    verbose=False
                )
                text = (res.text or "").strip()
                # Clean up output
                text = re.sub(r"^(NONE|无字幕|没有字幕|无)", "", text, flags=re.IGNORECASE).strip()
                # Clean up speaker prefix like "梁田：" or "张宁宁:" if present
                text = re.sub(r"^[\u4e00-\u9fa5]{2,6}[:：]\s*", "", text).strip()
                text = re.sub(r"[\r\n\t]", "", text)

                if text and len(text) >= 1:
                    extracted_cues.append({
                        "index": len(extracted_cues) + 1,
                        "start": round(itv["start"], 3),
                        "end": round(itv["end"], 3),
                        "duration": round(itv["end"] - itv["start"], 3),
                        "text": text,
                        "crop_image": str(crop_path),
                    })
                    print(f"  [{itv['start']:.2f}s -> {itv['end']:.2f}s]: {text}")
            except Exception as e:
                print(f"  ⚠️ Error OCRing interval #{idx}: {e}")

        # Clean and merge adjacent identical subtitle cues
        cleaned_cues = self._merge_consecutive_identical(extracted_cues)
        print(f"🎉 Successfully extracted {len(cleaned_cues)} on-screen ground-truth subtitles (merged from {len(extracted_cues)} raw samples)!")
        return cleaned_cues

    @staticmethod
    def _merge_consecutive_identical(cues: List[Dict[str, Any]], max_gap: float = 0.5) -> List[Dict[str, Any]]:
        """Merges consecutive cues with identical text."""
        if not cues:
            return []
        merged: List[Dict[str, Any]] = []
        cur = dict(cues[0])

        for nxt in cues[1:]:
            same_text = (nxt["text"] == cur["text"])
            gap = nxt["start"] - cur["end"]
            if same_text and gap <= max_gap:
                cur["end"] = nxt["end"]
                cur["duration"] = round(cur["end"] - cur["start"], 3)
            else:
                merged.append(cur)
                cur = dict(nxt)
        merged.append(cur)

        for i, c in enumerate(merged):
            c["index"] = i + 1
        return merged

    def export_srt(self, cues: List[Dict[str, Any]], output_srt: str) -> None:
        """Exports on-screen cues to standard SRT."""
        out_p = Path(output_srt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for c in cues:
            idx = c["index"]
            st = c["start"]
            et = c["end"]
            txt = c["text"]
            s_h, s_m, s_s = int(st // 3600), int((st % 3600) // 60), int(st % 60)
            s_ms = int(round((st - int(st)) * 1000))
            e_h, e_m, e_s = int(et // 3600), int((et % 3600) // 60), int(et % 60)
            e_ms = int(round((et - int(et)) * 1000))
            lines.append(str(idx))
            lines.append(f"{s_h:02d}:{s_m:02d}:{s_s:02d},{s_ms:03d} --> {e_h:02d}:{e_m:02d}:{e_s:02d},{e_ms:03d}")
            lines.append(txt)
            lines.append("")
        out_p.write_text("\n".join(lines), encoding="utf-8")
        print(f"💾 Saved on-screen ground truth SRT -> {output_srt}")
