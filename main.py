import argparse
import json
import sys
from pathlib import Path

from src.audio_separator import separate_vocals_and_bgm
from src.llm_backends import PROVIDER_PRESETS, LLMConfig, LLMError, make_backend
from src.media_engine import extract_audio, merge_video_subtitles
from src.srt_io import write_srt
from src.transcriber import transcribe_audio_mlx, transcribe_broadcast_stt, DEFAULT_PRIMARY_MODEL
from src.translator import build_brief, translate_segments


def process_video_pipeline(
    video_path: str,
    output_dir: str = "output",
    mode: str = "subbed",
    llm_config: LLMConfig | None = None,
    use_brief: bool = True,
    whisper_model: str = DEFAULT_PRIMARY_MODEL,
    stt_engine: str = "qwen3",
    isolate_vocals: bool = True,
    diarize: bool = False,
    ocr_assist: bool = False,
):
    """
    Unified media pipeline (a GUI can call this function directly):
    - mode="subbed": English subtitles embedded as a switchable track (original audio untouched)
    - mode="dubbed": isolates vocals via Demucs and prepares the background track for dubbing
    - llm_config picks the translation AI (see LLMConfig / PROVIDER_PRESETS in src/llm_backends.py)
    - ocr_assist: extracts burned-in subtitles from video frames to bias ASR glossary & entity recovery
    """
    llm_config = llm_config or LLMConfig(provider="deepseek")

    v_path = Path(video_path)
    if not v_path.exists():
        print(f"❌ Error: Video not found at '{video_path}'")
        print("💡 Put your video inside the 'videos/' folder and pass its relative path, e.g. --video videos/my_video.mp4")
        return 1

    stem = v_path.stem
    work_dir = Path(output_dir) / stem
    work_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"🚀 PROCESSING: {v_path.name}")
    print(f"🎯 Target Mode: {mode.upper()}   |   Translator: {llm_config.provider}")
    print(f"🎙️ STT Engine  : {stt_engine.upper()} (Model: {whisper_model if stt_engine == 'whisper' else stt_engine})")
    if ocr_assist:
        print("👁️ OCR Assist  : ENABLED (extracting on-screen burned-in subtitles)")
    print("=" * 65)

    # 1. Demux raw 16kHz audio
    extracted_wav = work_dir / f"{stem}_extracted.wav"
    print("\n[Step 1/4] Extracting 16kHz audio track...")
    extract_audio(str(v_path), str(extracted_wav))
    print(f"✅ Audio extracted -> {extracted_wav}")

    # 2. Source separation (prevents BGM/theme-song hallucinations during subtitling)
    if mode == "dubbed" or isolate_vocals:
        print("\n[Step 2/4] Isolating speech from background audio via Demucs...")
        stems = separate_vocals_and_bgm(str(extracted_wav), str(work_dir / "stems"))
        transcribe_target = stems["vocals"]
    else:
        print("\n[Step 2/4] Skipping vocal separation")
        transcribe_target = str(extracted_wav)

    # Optional: Visual Subtitle OCR Assist (biases vocabulary before STT)
    video_dossier_path = work_dir / f"{stem}_dossier.json"
    if ocr_assist:
        print("\n[OCR Assist] Scanning on-screen burned-in Chinese subtitles to bias STT vocabulary...")
        try:
            try:
                from src.stt.video_ocr import VideoSubtitleOCRExtractor, extract_visual_glossary
            except ImportError:
                from CleanProject.src.stt.video_ocr import VideoSubtitleOCRExtractor, extract_visual_glossary

            ocr_extractor = VideoSubtitleOCRExtractor()
            visual_cues = ocr_extractor.extract_subtitles_from_video(
                video_path=str(v_path),
                start_time=0.0,
                end_time=120.0,
                temp_dir=str(work_dir / "ocr_cache"),
            )
            ocr_extractor.unload()
            if visual_cues:
                ocr_srt = work_dir / f"{stem}_visual_gt.srt"
                ocr_extractor.export_srt(visual_cues, str(ocr_srt))
                glossary = extract_visual_glossary(visual_cues)
                dossier_data = {}
                if video_dossier_path.exists():
                    try:
                        dossier_data = json.loads(video_dossier_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                existing_terms = dossier_data.get("domain_glossary", {})
                for t in glossary["terms"]:
                    if t not in existing_terms:
                        existing_terms[t] = ""
                dossier_data["domain_glossary"] = existing_terms
                video_dossier_path.write_text(json.dumps(dossier_data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"✅ OCR Assist seeded {len(glossary['terms'])} Chinese terms & {len(glossary['names'])} names into {video_dossier_path}")
        except Exception as e:
            print(f"⚠️ OCR assist notice: {e}")

    # Optional: Audio-Visual Speaker Diarization (run BEFORE STT so turns guide chunking)
    turns_file = work_dir / "diar_v2" / "pyannote_turns.json"
    if diarize:
        print("\n[Diarization] Attributing speech to on-screen actors...")
        try:
            from src.diarizer import run_diarization_pipeline
        except ImportError:
            from CleanProject.src.diarizer import run_diarization_pipeline
        run_diarization_pipeline(
            video_path=str(v_path),
            vocals_wav=transcribe_target,
            output_dir=str(work_dir / "diar_v2"),
        )

    # 3. Transcribe speech on the Apple GPU (MLX)
    print(f"\n[Step 3/4] Transcribing dialogue with {stt_engine.upper()}...")
    if stt_engine == "whisper":
        segments = transcribe_audio_mlx(transcribe_target, model_repo=whisper_model)
    else:
        segments = transcribe_broadcast_stt(
            transcribe_target,
            orig_audio_path=str(extracted_wav),
            engine=stt_engine,
            dossier_path=str(video_dossier_path),
            turns_path=str(turns_file) if turns_file.exists() else None,
            roster_path=str(work_dir / "face_roster" / "confirmed_actor_roster.json"),
        )
    source_srt = work_dir / f"{stem}_source.srt"
    write_srt(segments, str(source_srt))
    print(f"✅ Initial source-language SRT ({len(segments)} lines) -> {source_srt}")

    # 4. Translate + DeepSeek Closed-Loop ASR Audit & Re-Subtitling
    print("\n[Step 4/4] Translating dialogue & running DeepSeek ASR Quality Audit...")
    try:
        with make_backend(llm_config) as llm:
            brief = build_brief(segments, llm) if use_brief else None
            if brief:
                brief_path = work_dir / f"{stem}_brief.json"
                brief_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"✅ Context dossier saved -> {brief_path}")
            translate_segments(
                segments,
                llm,
                brief=brief,
                audio_path=transcribe_target,
                enable_resubtitle_feedback=True,
                on_progress=lambda done, total: print(f"   {done}/{total} lines translated"),
            )
    except LLMError as e:
        print(f"❌ Translation failed: {e}")
        print(f"   Your transcription is safe in {source_srt}")
        return 1

    # Save finalized source SRT (with any DeepSeek-triggered Large-v3 corrections applied) and translated SRT
    write_srt(segments, str(source_srt))
    en_srt = work_dir / f"{stem}_translated.srt"
    write_srt(segments, str(en_srt), text_key="translated_text")
    print(f"✅ Finalized Source SRT -> {source_srt}")
    print(f"✅ Translated SRT       -> {en_srt}")

    # Final assembly
    if mode == "subbed":
        final_mp4 = work_dir / f"{stem}_subbed.mp4"
        print(f"\n🎬 Embedding subtitles -> {final_mp4}...")
        merge_video_subtitles(str(v_path), str(en_srt), str(final_mp4))
        print(f"🎉 Complete! Final video: {final_mp4}")
    elif mode == "dubbed":
        try:
            from src.dubber import dub_video_pipeline
        except ImportError:
            from CleanProject.src.dubber import dub_video_pipeline

        final_mp4 = work_dir / f"{stem}_dubbed.mp4"
        bgm_target = stems.get("no_vocals") if "stems" in locals() and stems else None
        print(f"\n🎙️ Synthesizing Multi-Voice English Dubbing -> {final_mp4}...")
        success = dub_video_pipeline(
            segments=segments,
            video_path=str(v_path),
            output_mp4=str(final_mp4),
            bgm_path=bgm_target,
            clips_dir=str(work_dir / "dubbing_clips"),
            force_resynth=True,
        )
        if success:
            print(f"🎉 Complete! Final Dubbed video: {final_mp4}")
        else:
            print(f"ℹ️ Prepared audio stems and translated timings in: {work_dir}")
    else:
        print(f"🎉 Complete! Output files saved to: {work_dir}")
    return 0


def resolve_local_model_path(custom_path: str = "") -> str:
    """Finds a GGUF model automatically from arguments, environment, or default locations."""
    if custom_path and Path(custom_path).is_file():
        return str(Path(custom_path).expanduser().resolve())
    import os
    env_path = os.environ.get("LOCAL_MODEL_PATH", "")
    if env_path and Path(env_path).is_file():
        return str(Path(env_path).expanduser().resolve())
    local_models = list(Path("models").glob("*.gguf"))
    if local_models:
        return str(local_models[0].resolve())
    # Check user home cache directories
    for cache_dir in [Path.home() / ".cache" / "lm-studio" / "models", Path.home() / ".cache" / "models"]:
        if cache_dir.exists():
            cached = list(cache_dir.glob("**/*.gguf"))
            if cached:
                return str(cached[0].resolve())
    raise FileNotFoundError("No .gguf model found! Pass --model-path or place a model inside models/")


def main():
    parser = argparse.ArgumentParser(description="AI video subtitling & dubbing studio")
    parser.add_argument("--video", type=str, default="videos/sample.mp4", help="Path to the input video file")
    parser.add_argument("--mode", choices=["subbed", "dubbed"], default="subbed",
                        help="'subbed' = subtitles, 'dubbed' = prepare voice replacement")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--whisper-model", type=str, default=DEFAULT_PRIMARY_MODEL,
                        help="Primary MLX Whisper model repo (when using --stt-engine whisper)")
    parser.add_argument("--stt-engine", choices=["qwen3", "whisper", "funasr", "sensevoice"], default="qwen3",
                        help="Speech-to-Text engine (default: qwen3)")
    parser.add_argument("--diarize", action="store_true",
                        help="Attribute speech to on-screen actors via audio-visual diarization (pyannote + face tracking + ASD)")
    parser.add_argument("--ocr-assist", action="store_true",
                        help="Extract burned-in Chinese subtitles from video frames to assist STT entity recognition")

    llm = parser.add_argument_group("translation AI")
    llm.add_argument("--local", action="store_true",
                     help="Run locally on Apple GPU via llama.cpp (sets provider to llama_cpp)")
    llm.add_argument("--provider", choices=list(PROVIDER_PRESETS), default="deepseek",
                     help="Which AI translates. API keys are read from environment variables "
                          "(DEEPSEEK_API_KEY, OPENROUTER_API_KEY, ...), never from the command line.")
    llm.add_argument("--model", default="", help="Model name (each provider has a default where one is known)")
    llm.add_argument("--base-url", default="", help="API address (required for custom_* providers)")
    llm.add_argument("--model-path", default="", help="Path to a .gguf file (auto-detected if omitted with --local)")
    llm.add_argument("--no-brief", action="store_true",
                     help="Skip the first pass where the model reads the transcript and takes notes")
    args = parser.parse_args()

    provider = "llama_cpp" if args.local else args.provider
    model_path = resolve_local_model_path(args.model_path) if provider == "llama_cpp" else ""

    config = LLMConfig(provider=provider, model=args.model,
                       base_url=args.base_url, model_path=model_path)
    sys.exit(process_video_pipeline(
        video_path=args.video, output_dir=args.output, mode=args.mode,
        llm_config=config, use_brief=not args.no_brief, whisper_model=args.whisper_model,
        stt_engine=args.stt_engine, diarize=args.diarize, ocr_assist=args.ocr_assist))


if __name__ == "__main__":
    main()
