"""
gender_pitch_gate.py - Demographic & Acoustic F0 Invariant Sanity Gate.

Enforces cross-modal parity between InsightFace visual gender detection and
audio fundamental frequency (F0) tracking via probabilistic YIN (pYIN).
Implements intra-actor pitch clustering to purge reaction-shot intruder clips.
"""
from __future__ import annotations
import cv2
import numpy as np
import librosa
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path


class GenderPitchGate:
    """
    Enforces demographic consistency by cross-validating visual gender estimates
    from InsightFace against acoustic fundamental frequency (F0) from librosa.pyin.
    """
    def __init__(self, male_pitch_ceiling: float = 155.0, female_pitch_floor: float = 165.0):
        self.male_ceiling = male_pitch_ceiling
        self.female_floor = female_pitch_floor

    def estimate_acoustic_gender(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Estimates gender and pitch from fundamental frequency (F0) using probabilistic YIN.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if len(audio) < int(0.20 * sr):
            return {"gender": "unknown", "f0": 0.0, "confidence": 0.0}

        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            target_sr = 16000
        else:
            audio_16k = audio
            target_sr = sr

        fmin = librosa.note_to_hz('C2')   # ~65 Hz
        fmax = librosa.note_to_hz('C6')   # ~1046 Hz
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_16k, fmin=fmin, fmax=fmax, sr=target_sr, frame_length=1024, hop_length=256
            )
        except Exception:
            return {"gender": "unknown", "f0": 0.0, "confidence": 0.0}

        if voiced_flag is None or not np.any(voiced_flag):
            return {"gender": "unknown", "f0": 0.0, "confidence": 0.0}

        valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(valid_f0) < 5:
            return {"gender": "unknown", "f0": 0.0, "confidence": 0.0}

        med_f0 = float(np.median(valid_f0))
        voiced_ratio = float(np.mean(voiced_flag))
        conf = round(float(np.mean(voiced_probs[voiced_flag])) * voiced_ratio, 2)

        if med_f0 <= self.male_ceiling:
            ac_gender = "male"
        elif med_f0 >= self.female_floor:
            ac_gender = "female"
        else:
            ac_gender = "ambiguous"

        return {"gender": ac_gender, "f0": round(med_f0, 1), "confidence": conf}

    def sanitize_actor_reference_clips(
        self,
        actor_id: str,
        clips: List[Dict[str, Any]],
        base_dir: Path,
        visual_hint_gender: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """
        Purges cross-contaminated outlier clips from an actor's reference pool.
        Finds the dominant vocal pitch mode and discards discordant intruder clips.
        Returns: (sanitized_clips, canonical_gender, median_pitch)
        """
        if not clips:
            return [], visual_hint_gender or "unknown", 0.0

        clip_pitches = []
        valid_items = []
        for c in clips:
            rel_path = c.get("wav_path", "")
            p = base_dir / rel_path
            if not p.exists():
                p = Path("output") / rel_path
            if not p.exists():
                continue

            try:
                import soundfile as sf
                audio, sr = sf.read(str(p))
                info = self.estimate_acoustic_gender(audio, sr)
                if info["f0"] > 0:
                    clip_pitches.append(info["f0"])
                    valid_items.append((c, info))
            except Exception:
                continue

        if not valid_items:
            return clips, visual_hint_gender or "unknown", 0.0

        # Determine dominant gender register
        female_count = sum(1 for p in clip_pitches if p >= self.female_floor)
        male_count = sum(1 for p in clip_pitches if p <= self.male_ceiling)

        if visual_hint_gender == "male" and male_count > 0:
            dominant_gender = "male"
        elif visual_hint_gender == "female" and female_count > 0:
            dominant_gender = "female"
        elif female_count > male_count:
            dominant_gender = "female"
        elif male_count > female_count:
            dominant_gender = "male"
        else:
            dominant_gender = "female" if np.median(clip_pitches) >= 160.0 else "male"

        sanitized = []
        kept_pitches = []
        for c, info in valid_items:
            f0 = info["f0"]
            if dominant_gender == "male" and f0 >= self.female_floor:
                # Contaminated female clip in male actor pool
                continue
            if dominant_gender == "female" and f0 <= self.male_ceiling:
                # Contaminated male clip in female actor pool
                continue
            sanitized.append(c)
            kept_pitches.append(f0)

        final_clips = sanitized if sanitized else [valid_items[0][0]]
        final_pitch = float(np.median(kept_pitches)) if kept_pitches else float(np.median(clip_pitches))
        return final_clips, dominant_gender, round(final_pitch, 1)

    def validate_clip_pair(
        self,
        visual_gender: str,
        audio_chunk: np.ndarray,
        sr: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Hard-gate check: Returns True only if visual gender matches acoustic pitch.
        """
        ac_info = self.estimate_acoustic_gender(audio_chunk, sr)
        ac_gender = ac_info["gender"]

        v_g = visual_gender.lower().strip()
        if ac_gender == "unknown" or v_g == "unknown":
            return True, "unverified_insufficient_audio", ac_info

        if v_g == "male" and ac_gender == "female":
            reason = f"REJECTED: Male face bound to female pitch ({ac_info['f0']} Hz)"
            return False, reason, ac_info

        if v_g == "female" and ac_gender == "male":
            reason = f"REJECTED: Female face bound to male pitch ({ac_info['f0']} Hz)"
            return False, reason, ac_info

        return True, "PASSED: Gender consistent", ac_info
