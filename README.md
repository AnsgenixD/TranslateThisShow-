# TranslateThisShow

An end-to-end subtitling, speaker diarization, and multi-speaker dubbing pipeline for Chinese-to-English video, built to run natively on Apple Silicon (`mlx`).

## Demo

Sample output generated on Apple Silicon using `Qwen3-ASR`, `pyannote` + `CAM++` speaker attribution, DeepSeek translation, `KittenTTS` synthesis, and `demucs-mlx` background audio preservation:

https://github.com/user-attachments/assets/28220f1c-5779-4e39-aa65-e5d9c5fb0a5b

### Pipeline Architecture

1. **Stem Separation (`src/audio_separator.py`)**
   Splits audio into isolated vocal and accompaniment stems using `demucs-mlx`. Transcribing the vocal stem prevents background music from causing ASR hallucinations, while preserving the original BGM track for dubbed output.
2. **Speech Recognition & Alignment (`src/stt/`, `src/transcriber.py`)**
   Supports `Qwen3-ASR`, `SenseVoiceSmall`, `FunASR`, and `mlx-whisper`. Uses moving-average energy chunking, character-level forced alignment (`Qwen3-ForcedAligner-0.6B-8bit`), cross-boundary orphan stitching, and toneless-pinyin homophone normalization (`src/stt/name_fixer.py`).
3. **Speaker Diarization & Attribution (`src/diarizer.py`, `src/diarization_v2/`)**
   Combines `pyannote/speaker-diarization-community-1` turn clustering, Alibaba 3D-Speaker `CAM++` 192-D voiceprint embeddings, `librosa.yin` pitch ($F_0$) gender estimation, and grammatical sentence-continuity locking so speakers do not switch mid-sentence. Optional `--diarize` adds InsightFace tracking and lip-motion active speaker detection.
4. **Context-Aware Translation (`src/translator.py`, `src/llm_backends.py`)**
   Runs a two-pass workflow: first building a show dossier (`*_brief.json`) of character names, terminology, and ASR phonetic repairs, then translating batches with surrounding context. Supports any OpenAI-compatible API endpoint (DeepSeek, OpenRouter, OpenAI, Groq, or custom endpoints) as well as offline local translation on Apple Silicon using `mlx-community` models (`mlx-lm`).
5. **Dubbing & Muxing (`src/dubber.py`, `src/media_engine.py`)**
   Groups consecutive same-speaker clauses into complete sentences, synthesizes speech via local `KittenTTS` (24 kHz ONNX) or cloud `edge-tts`, time-stretches clips to fit slot headroom, mixes with the Demucs accompaniment stem, and muxes audio and switchable subtitles via `ffmpeg`.

---

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.10+
- `ffmpeg` (`brew install ffmpeg`)

### Installation & Interactive Setup (`setup_and_debug.py`)

```bash
git clone https://github.com/AnsgenixD/TranslateThisShow-.git
cd TranslateThisShow-
python setup_and_debug.py
```

`setup_and_debug.py` (also accessible via `python main.py --setup` or triggered automatically if `main.py` encounters a missing dependency, model weight, or API key error) performs four tasks:

1. **System & Code Diagnostics**: Verifies `ffmpeg`, Apple Metal GPU acceleration, all Python packages, `espeak-ng` library bindings for `KittenTTS`, Alibaba `3D-Speaker` `CAM++` weights, and internal module imports, installing missing dependencies automatically.
2. **Secrets & Translation Backend Configuration (`.env`)**: Prompts for optional API keys (`DEEPSEEK_API_KEY`, `HF_TOKEN`, or any custom OpenAI-compatible endpoint URL/key) or configures offline local translation.
3. **Mac Hardware Profiler & `mlx-community` Model Advisor**: Detects your Apple Silicon chip and Unified Memory capacity (8 GB, 16 GB, 24 GB, 32 GB+) and recommends `mlx-community` translation LLMs with estimated memory usage and token generation speed on your Mac.
4. **Model Weight Installer**: Checks and downloads required `mlx-community` STT (`Qwen3-ASR`, `Whisper-large-v3`, `SenseVoiceSmall`), forced alignment (`Qwen3-ForcedAligner`), VLM OCR (`Qwen3-VL`, `Qwen2.5-VL`), `KittenTTS`, and `CAM++` weights.

To run a non-interactive environment and model readiness check at any time:

```bash
python setup_and_debug.py --check
```

---

## Usage

Place input videos in `videos/` (or pass any file path to `--video`). Intermediate files and final videos are written to `output/<video_name>/` by default, or you can pass `--output videos` to write outputs directly to `videos/`.

### Subtitling (`--mode subbed`)

```bash
export DEEPSEEK_API_KEY="your_api_key"

# Default (Qwen3-ASR)
python main.py --video videos/my_video.mp4 --mode subbed

# Alternative ASR backends
python main.py --video videos/my_video.mp4 --mode subbed --stt-engine whisper
python main.py --video videos/my_video.mp4 --mode subbed --stt-engine sensevoice
```

### Multi-Speaker Dubbing (`--mode dubbed`)

```bash
# On-device KittenTTS (local CPU/ONNX, no cloud TTS calls)
python main.py --video videos/my_video.mp4 --mode dubbed --tts-engine kittentts

# Microsoft Neural voices (cloud)
python main.py --video videos/my_video.mp4 --mode dubbed --tts-engine edge-tts

# With face tracking & lip active speaker detection
python main.py --video videos/my_video.mp4 --mode dubbed --diarize
```

### Burned-In Subtitle OCR Assist (`--ocr-assist`)

Samples video frames to extract on-screen Chinese subtitles and seed character names and domain terms before transcription:

```bash
python main.py --video videos/my_video.mp4 --mode subbed --ocr-assist
```

### Local Offline Translation (`mlx-community`)

Run translation completely offline on Apple Silicon Unified Memory using quantized models from `mlx-community`:

```bash
# Uses default mlx-community/Qwen2.5-7B-Instruct-4bit (or the model configured in .env)
python main.py --video videos/my_video.mp4 --mode dubbed --local

# Specify a particular mlx-community model for your Mac's RAM tier
python main.py --video videos/my_video.mp4 --mode dubbed --local --model mlx-community/Qwen3-8B-4bit
```

---

## CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--setup`, `--doctor` | `False` | Run the interactive installer, hardware profiler, secrets manager, and debugger. |
| `--video` | `videos/sample.mp4` | Path to the input video file. |
| `--mode` | `subbed` | `subbed` embeds a switchable English subtitle track; `dubbed` generates a multi-voice dubbed audio track plus subtitles. |
| `--output` | `output` | Root directory for generated stems, SRTs, and output videos. |
| `--stt-engine` | `qwen3` | Speech recognition backend: `qwen3`, `whisper`, `funasr`, `sensevoice`. |
| `--whisper-model` | `mlx-community/whisper-large-v3-mlx` | Model repo used when `--stt-engine whisper` is selected. |
| `--tts-engine` | `kittentts` | Dubbing synthesis backend: `kittentts` (local ONNX) or `edge-tts` (cloud). |
| `--diarize` | `False` | Enable face tracking and active speaker detection in addition to acoustic clustering. |
| `--ocr-assist` | `False` | Scan video frames for burned-in Chinese subtitles to seed the glossary. |
| `--provider` | `deepseek` | Translation backend: `deepseek`, `mlx`, `openai`, `anthropic`, `openrouter`, `groq`, `custom_openai`, `llama_cpp`. |
| `--local` | `False` | Run offline on Apple Silicon using `mlx-community` models (`mlx`), or `llama_cpp` if `--model-path` is given. |
| `--model` | `""` | Override the default model name (e.g. `mlx-community/Qwen3-8B-4bit`). |
| `--base-url` | `""` | Custom API endpoint (for `custom_openai` / `custom_anthropic`). |
| `--model-path` | `""` | Path to a local `.gguf` file when using `llama.cpp`. |
| `--no-brief` | `False` | Skip the first-pass dossier/glossary extraction step. |

---

## Output Structure

Running `python main.py --video videos/my_video.mp4 --mode dubbed` generates:

```text
output/my_video/
├── my_video_extracted.wav      # Demuxed 16kHz audio
├── stems/
│   ├── vocals.wav              # Isolated speech stem (Demucs)
│   └── accompaniment.wav       # Isolated background audio / BGM stem
├── diar_v2/
│   ├── pyannote.rttm           # Standard speaker diarization RTTM
│   ├── pyannote_exclusive.rttm # Non-overlapping speaker turns
│   ├── pyannote_turns.json     # Parsed speaker turns
│   └── vocals_16k.wav          # 16kHz mono vocal track
├── dubbing_clips/
│   └── line_0001.wav ...       # Synthesized speech clips per utterance
├── my_video_dossier.json       # Character registry and domain glossary
├── my_video_brief.json         # Contextual translation brief
├── my_video_source.srt         # Source-language subtitles
├── my_video_translated.srt     # English subtitles
├── my_video_dubbed.wav         # Mixed speech + BGM master audio
└── my_video_dubbed.mp4         # Final dubbed video with switchable subtitles
```

---

## License

MIT License. See [LICENSE](LICENSE).
