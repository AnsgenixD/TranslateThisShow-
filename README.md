# TranslateThisShow

End-to-end subtitling, speaker diarization, and multi-speaker dubbing pipeline for long-form video, built for Apple Silicon (`mlx`).

## Pipeline Overview

1. **Stem Separation (`src/audio_separator.py`)**
   Splits raw audio into isolated vocal and background (BGM/ambience) stems via `demucs-mlx`. This prevents background music from triggering ASR hallucinations during transcription and preserves the original background track for dubbing.
2. **Speech-to-Text (`src/stt/`, `src/transcriber.py`)**
   - Supports multiple Apple Silicon ASR backends (`mlx-whisper`, `Qwen3-ASR`, `SenseVoice`, `FunASR`).
   - Performs a two-pass entity discovery step where the LLM extracts character names and domain terminology from the initial transcript (`show_dossier.json`), followed by toneless-pinyin homophone correction (`src/stt/name_fixer.py`).
   - Optional character-level forced alignment (`src/stt/aligner.py`) and on-screen subtitle OCR (`src/stt/video_ocr.py`).
3. **Speaker Diarization (`src/diarization_v2/`, `src/diarizer.py`)**
   Combines acoustic speaker turn clustering (`pyannote.audio`), face tracking (`insightface`), lip-motion active speaker detection, and 3D-Speaker CAM++ voiceprint embeddings via bipartite Hungarian matching.
4. **Translation (`src/translator.py`, `src/llm_backends.py`)**
   Context-aware translation using either cloud APIs (DeepSeek, OpenAI, OpenRouter, Groq, Anthropic) or local `.gguf` models via `llama.cpp`.
5. **Dubbing & Muxing (`src/dubber.py`, `src/media_engine.py`)**
   Synthesizes speaker-matched dialogue tracks (`edge-tts`), adjusts speaking rate to fit timing windows without truncating words, mixes against the isolated BGM stem, and muxes the final audio or subtitle track via `ffmpeg`.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- `ffmpeg` (`brew install ffmpeg`)

### Core Installation

```bash
git clone https://github.com/AnsgenixD/TranslateThisShow-.git
cd TranslateThisShow-
pip install -r requirements.txt
```

### Optional Packages

The pipeline runs out of the box with `mlx-whisper`. Optional components are loaded dynamically if installed, and skipped with a fallback if not present:

- **Qwen3-ASR / SenseVoice / Forced Alignment**:
  ```bash
  pip install mlx-audio
  ```
  Supported models (downloaded automatically on first use or via `huggingface-cli`):
  - `mlx-community/Qwen3-ASR-0.6B-8bit` (default when `--stt-engine qwen3` is selected)
  - `mlx-community/Qwen3-ASR-1.7B-8bit`
  - `mlx-community/Qwen3-ForcedAligner-0.6B-8bit`
  - `mlx-community/SenseVoiceSmall`

- **On-Screen Subtitle OCR & VLM Speaker Arbitration**:
  ```bash
  pip install mlx-vlm opencv-python Pillow
  ```
  Recommended vision models: `mlx-community/Qwen3-VL-4B-Instruct-4bit` or `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`.

- **Audio-Visual Speaker Diarization**:
  ```bash
  pip install pyannote.audio insightface opencv-python
  export HF_TOKEN="your_huggingface_token"
  ```

## Usage

Place input videos in `videos/` (or pass any file path to `--video`). Generated artifacts are written to `output/<video_name>/`.

### Generate Subtitled Video

```bash
export DEEPSEEK_API_KEY="your_api_key"
python main.py --video videos/sample.mp4 --mode subbed
```

Select a specific ASR engine:
```bash
python main.py --video videos/sample.mp4 --mode subbed --stt-engine whisper
python main.py --video videos/sample.mp4 --mode subbed --stt-engine qwen3
```

### Generate Dubbed Video

```bash
python main.py --video videos/sample.mp4 --mode dubbed
```

Include audio-visual speaker diarization for per-speaker voice assignment:
```bash
python main.py --video videos/sample.mp4 --mode dubbed --diarize
```

### Local Offline Translation (`llama.cpp`)

Place a `.gguf` model in `models/` (or pass `--model-path`):
```bash
python main.py --video videos/sample.mp4 --mode subbed --local --model-path models/model.gguf
```

## Repository Structure

```text
.
├── main.py                  # CLI entry point
├── requirements.txt         # Core Python dependencies
├── benchmarks/
│   └── stt_bench.py         # CER, entity recall, and RTF evaluation script
├── src/
│   ├── audio_separator.py   # Demucs-MLX stem separation
│   ├── diarizer.py          # Diarization pipeline wrapper
│   ├── dubber.py            # Multi-voice TTS synthesis and BGM mixing
│   ├── llm_backends.py      # OpenAI/Anthropic/DeepSeek/llama.cpp client adapters
│   ├── media_engine.py      # FFmpeg audio extraction and subtitle muxing
│   ├── srt_io.py            # SRT read/write utilities
│   ├── transcriber.py       # ASR entry point
│   ├── translator.py        # Two-pass dialogue translator
│   ├── stt/                 # Modular STT, forced alignment, pinyin fixer, video OCR
│   └── diarization_v2/      # Acoustic clustering, face tracking, ASD, Hungarian matching
├── videos/                  # Input video directory
└── output/                  # Output directory
```

## License

MIT License. See [LICENSE](LICENSE).
