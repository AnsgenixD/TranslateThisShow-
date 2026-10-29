import argparse
import json
import sys
from pathlib import Path

from src.audio_separator import separate_vocals_and_bgm
from src.llm_backends import PROVIDER_PRESETS, LLMConfig, LLMError, make_backend
from src.media_engine import extract_audio, merge_video_subtitles
from src.srt_io import write_srt
from src.transcriber import transcribe_audio_mlx
from src.translator import build_brief, translate_segments


def process_video_pipeline(
    video_path: str,
    output_dir: str = "output",
    mode: str = "subbed",
    llm_config: LLMConfig | None = None,
    use_brief: bool = True,
):
    """
    Unified media pipeline (a GUI can call this function directly):
    - mode="subbed": English subtitles embedded as a switchable track (original audio untouched)
    - mode="dubbed": isolates vocals via Demucs and prepares the background track for dubbing
    - llm_config picks the translation AI (see LLMConfig / PROVIDER_PRESETS in src/llm_backends.py)
    """
    llm_config = llm_config or LLMConfig()

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
    print("=" * 65)

    # 1. Demux raw 16kHz audio
    extracted_wav = work_dir / f"{stem}_extracted.wav"
    print("\n[Step 1/4] Extracting 16kHz audio track...")
    extract_audio(str(v_path), str(extracted_wav))
    print(f"✅ Audio extracted -> {extracted_wav}")

    # 2. Source separation (only needed for dubbing)
    if mode == "dubbed":
        print("\n[Step 2/4] Isolating speech from background audio via Demucs-MLX...")
        stems = separate_vocals_and_bgm(str(extracted_wav), str(work_dir / "stems"))
        transcribe_target = stems["vocals"]
    else:
        print("\n[Step 2/4] Skipping vocal separation (subbed mode keeps the original audio)")
        transcribe_target = str(extracted_wav)

    # 3. Transcribe speech on the Apple GPU (MLX)
    print("\n[Step 3/4] Transcribing dialogue with mlx-whisper...")
    segments = transcribe_audio_mlx(transcribe_target)
    source_srt = work_dir / f"{stem}_source.srt"
    write_srt(segments, str(source_srt))
    print(f"✅ Source-language SRT -> {source_srt}")

    # 4. Translate with the chosen LLM
    print("\n[Step 4/4] Translating dialogue...")
    try:
        with make_backend(llm_config) as llm:
            brief = build_brief(segments, llm) if use_brief else None
            if brief:
                brief_path = work_dir / f"{stem}_brief.json"
                brief_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"✅ The model's notes on this video -> {brief_path}")
            translate_segments(
                segments, llm, brief=brief,
                on_progress=lambda done, total: print(f"   {done}/{total} lines translated"))
    except LLMError as e:
        print(f"❌ Translation failed: {e}")
        print(f"   Your transcription is safe in {source_srt}")
        return 1

    en_srt = work_dir / f"{stem}_translated.srt"
    write_srt(segments, str(en_srt), text_key="translated_text")
    print(f"✅ Translated SRT -> {en_srt}")

    # Final assembly
    if mode == "subbed":
        final_mp4 = work_dir / f"{stem}_subbed.mp4"
        print(f"\n🎬 Embedding subtitles -> {final_mp4}...")
        merge_video_subtitles(str(v_path), str(en_srt), str(final_mp4))
        print(f"🎉 Complete! Final video: {final_mp4}")
    else:
        print(f"🎉 Complete! Audio stems and translated timings are in: {work_dir}")
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
    known_default = Path("/Users/ansgenix/Desktop/DO THINGS INSIDE THIS DIRECTORY/avatar/models/llm/Qwen3-8B-Q4_K_M.gguf")
    if known_default.is_file():
        return str(known_default)
    raise FileNotFoundError("No .gguf model found! Pass --model-path or place a model inside models/")


def main():
    parser = argparse.ArgumentParser(description="AI video subtitling & dubbing studio")
    parser.add_argument("--video", type=str, default="videos/sample.mp4", help="Path to the input video file")
    parser.add_argument("--mode", choices=["subbed", "dubbed"], default="subbed",
                        help="'subbed' = subtitles, 'dubbed' = prepare voice replacement")
    parser.add_argument("--output", type=str, default="output", help="Output directory")

    llm = parser.add_argument_group("translation AI")
    llm.add_argument("--local", action="store_true",
                     help="Run locally on Apple GPU via llama.cpp (sets provider to llama_cpp)")
    llm.add_argument("--provider", choices=list(PROVIDER_PRESETS), default="openrouter",
                     help="Which AI translates. API keys are read from environment variables "
                          "(OPENROUTER_API_KEY, DEEPSEEK_API_KEY, ...), never from the command line.")
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
        llm_config=config, use_brief=not args.no_brief))


if __name__ == "__main__":
    main()
