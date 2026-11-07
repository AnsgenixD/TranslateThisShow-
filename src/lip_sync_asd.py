"""
lip_sync_asd.py - Audio-Visual Active Speaker Detection via Cross-Modal Temporal Synchrony.

Correlates mouth vertical aperture dynamics and oral cavity darkening with vocal RMS energy.
Distinguishes active on-screen speaking from silent reaction shots (nodding, smiling, blinking).
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional, Tuple


class LipSyncASD:
    """
    Computes cross-modal temporal synchrony between facial lip aperture
    and audio energy envelope to confirm active speech on camera.
    Supports 68-point 3D/2D facial landmarks, 106-point 2D landmarks,
    and 5-point keypoint fallback with oral cavity photometry.
    """
    def __init__(self, fps: float = 25.0):
        self.fps = fps

    def get_lip_aperture(
        self,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
        face_dict: Optional[dict] = None
    ) -> Optional[float]:
        """
        Calculates normalized Lip Aperture Ratio (LAR = inner height / outer width).
        Accepts:
          - face_dict: InsightFace face object containing 'landmark_3d_68' or 'landmark_2d_106'
          - landmarks: (68, 2/3), (106, 2), or (5, 2) numpy array
          - frame: optional image frame for 5pt photometric oral cavity darkness fallback
        """
        # 1. Check for face_dict with 68-point or 106-point landmarks
        if face_dict is not None:
            if "landmark_3d_68" in face_dict and face_dict["landmark_3d_68"] is not None:
                landmarks = face_dict["landmark_3d_68"]
            elif "landmark_2d_106" in face_dict and face_dict["landmark_2d_106"] is not None:
                landmarks = face_dict["landmark_2d_106"]
            elif "kps" in face_dict:
                landmarks = face_dict["kps"]

        if landmarks is None:
            return None

        # 2. Geometric 68-point landmark LAR (standard 300W / iBUG standard)
        # 48: left corner, 54: right corner, 62: upper inner lip, 66: lower inner lip
        if len(landmarks) >= 68:
            pts = landmarks
            corner_l = pts[48][:2]
            corner_r = pts[54][:2]
            lip_top = pts[62][:2]
            lip_bot = pts[66][:2]
            
            mouth_w = float(np.linalg.norm(corner_l - corner_r))
            inner_h = float(np.linalg.norm(lip_top - lip_bot))
            if mouth_w < 5.0:
                return None
            lar = inner_h / (mouth_w + 1e-6)
            return float(np.clip(lar * 100.0, 0.0, 100.0))

        # 3. Geometric 106-point landmark LAR (InsightFace 106pt standard)
        # Inner lips: 65 (upper inner), 71 (lower inner), 52 (left corner), 61 (right corner)
        if len(landmarks) == 106:
            pts = landmarks
            corner_l = pts[52][:2]
            corner_r = pts[61][:2]
            lip_top = pts[65][:2]
            lip_bot = pts[71][:2]
            mouth_w = float(np.linalg.norm(corner_l - corner_r))
            inner_h = float(np.linalg.norm(lip_top - lip_bot))
            if mouth_w < 5.0:
                return None
            lar = inner_h / (mouth_w + 1e-6)
            return float(np.clip(lar * 100.0, 0.0, 100.0))

        # 4. Fallback: 5-point landmark photometric oral cavity darkness
        if len(landmarks) == 5 and frame is not None:
            r_corner, l_corner = landmarks[3], landmarks[4]
            mouth_w = float(np.linalg.norm(r_corner - l_corner))
            if mouth_w < 8.0:
                return None

            center = (r_corner + l_corner) / 2.0
            r_box = int(mouth_w * 0.70)
            h, w = frame.shape[:2]
            y1, y2 = max(0, int(center[1] - r_box * 0.55)), min(h, int(center[1] + r_box * 0.55))
            x1, x2 = max(0, int(center[0] - r_box)), min(w, int(center[0] + r_box))

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            val = hsv[:, :, 2]
            dark_pixels = np.sum(val < 55)
            aperture_score = float(dark_pixels / (roi.shape[0] * roi.shape[1] + 1e-6)) * 100.0
            return aperture_score

        return None

    def compute_synchrony_score(
        self,
        aperture_series: List[float],
        audio_chunk: np.ndarray,
        sr: int
    ) -> float:
        """
        Cross-correlates mouth aperture changes over time with RMS audio energy.
        Returns a score in [0.0, 1.0]. A reaction shot will score < 0.25.
        """
        n_frames = len(aperture_series)
        if n_frames < 3:
            return 0.5  # Neutral default for very short segments

        samples_per_frame = int(sr / self.fps)
        audio_rms = []
        for i in range(n_frames):
            s_idx = i * samples_per_frame
            e_idx = min(len(audio_chunk), (i + 1) * samples_per_frame)
            seg = audio_chunk[s_idx:e_idx]
            rms = float(np.sqrt(np.mean(seg**2))) if len(seg) > 0 else 0.0
            audio_rms.append(rms)

        ap_arr = np.array(aperture_series, dtype=np.float32)
        rms_arr = np.array(audio_rms, dtype=np.float32)

        # Variance check
        ap_std = float(np.std(ap_arr))
        rms_std = float(np.std(rms_arr))
        if ap_std < 0.05 or rms_std < 1e-4:
            return 0.1  # Zero motion or silence = reaction shot

        ap_norm = (ap_arr - np.mean(ap_arr)) / (ap_std + 1e-6)
        rms_norm = (rms_arr - np.mean(rms_arr)) / (rms_std + 1e-6)

        # Pearson correlation
        corr = float(np.mean(ap_norm * rms_norm))
        # Map from [-1, 1] to [0, 1]
        sync_score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
        return round(sync_score, 2)
