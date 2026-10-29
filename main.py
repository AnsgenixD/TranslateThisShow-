import sys
import argparse
from pathlib import Path
from src.media_engine import extract_audio, merge_video_subtitles
from src.audio_separator import separate_vocals_and_bgm
from src.transcriber import transcribe_audio_mlx, write_srt
from src.translator import translate_dialogue_segments

def process_video_pipeline(video_path: str, output_dir: str = "output", mode: str = "subbed"):
    """
    Unified Media Pipeline:
    - mode="subbed": Anime-style English subtitles burned/embedded on video (original speech & audio intact)
    - mode="dubbed": Isolates vocals via Demucs, prepares accompaniment BGM track for speech replacement
    """
    v_path = Path(video_path)
    if not v_path.exists():
        print(f"❌ Error: Video not found at '{video_path}'")
        print("💡 Place your video file inside the 'videos/' folder and pass its relative path.")
        return

    out_root = Path(output_dir)
    stem = v_path.stem
    work_dir = out_root / stem
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print(f"🚀 PROCESSING: {v_path.name}")
    print(f"🎯 Target Mode: {mode.upper()}")
    print("=" * 65)

    # 1. Demux raw 16kHz audio
    extracted_wav = work_dir / f"{stem}_extracted.wav"
    print(f"\n[Step 1/4] Extracting 16kHz audio track...")
    extract_audio(str(v_path), str(extracted_wav))
    print(f"✅ Audio extracted -> {extracted_wav}")

    # 2. Source Separation (only required if dubbing)
    if mode == "dubbed":
        print(f"\n[Step 2/4] Isolating speech from BGM via Demucs-MLX...")
        stems = separate_vocals_and_bgm(str(extracted_wav), str(work_dir / "stems"))
        transcribe_target = stems["vocals"]
    else:
        print(f"\n[Step 2/4] Skipping vocal separation (Subbed mode preserves original audio)")
        transcribe_target = str(extracted_wav)

    # 3. Transcribe Chinese Speech on Apple GPU (MLX)
    print(f"\n[Step 3/4] Transcribing dialogue with mlx-whisper...")
    segments = transcribe_audio_mlx(transcribe_target)
    zh_srt = work_dir / f"{stem}_chinese.srt"
    write_srt(segments, str(zh_srt))
    print(f"✅ Chinese SRT generated -> {zh_srt}")

    # 4. Context-Aware Translation
    print(f"\n[Step 4/4] Translating dialogue (ZH -> EN)...")
    translated_segments = translate_dialogue_segments(segments, mode="local")
    en_srt = work_dir / f"{stem}_english.srt"
    write_srt(translated_segments, str(en_srt))
    print(f"✅ English SRT generated -> {en_srt}")

    # Final Assembly (Subbed mode generates final subtitled MP4)
    if mode == "subbed":
        final_mp4 = work_dir / f"{stem}_subbed.mp4"
        print(f"\n🎬 Rendering Subtitled Video -> {final_mp4}...")
        merge_video_subtitles(str(v_path), str(en_srt), str(final_mp4))
        print(f"🎉 Complete! Final video saved at: {final_mp4}")
    else:
        print(f"🎉 Complete! Audio stems and English timings ready for TTS synthesis in: {work_dir}")

def main():
    parser = argparse.ArgumentParser(description="AI Video Subtitling & Dubbing Studio")
    parser.add_argument("--video", type=str, default="videos/sample.mp4", help="Path to input video file")
    parser.add_argument("--mode", type=str, choices=["subbed", "dubbed"], default="subbed", help="Target mode: 'subbed' (subtitles) or 'dubbed' (voice replacement)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    process_video_pipeline(video_path=args.video, output_dir=args.output, mode=args.mode)

if __name__ == "__main__":
    main()
