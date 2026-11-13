# TranslateThisShow

An end-to-end subtitling, speaker diarization, and multi-speaker dubbing pipeline for Chinese-to-English video, built to run natively on Apple Silicon (`mlx`).

## Demo

Sample output generated on Apple Silicon using `Qwen3-ASR`, `pyannote` + `CAM++` speaker attribution, DeepSeek translation, `KittenTTS` synthesis, and `demucs-mlx` background audio preservation:

[https://github.com/user-attachments/assets/28220f1c-5779-4e39-aa65-e5d9c5fb0a5b](https://github.com/user-attachments/assets/43f8a44b-2c72-473f-9cd9-182d26bd90d5)

### Pipeline Architecture

1. **Stem Separation (`src/audio_separator.py`)**
   Splits audio into isolated vocal and accompaniment stems using `demucs-mlx`. Transcribing the vocal stem prevents background music from causing ASR hallucinations, while preserving the original BGM track for dubbed output.
2. **Speech Recognition & Alignment (`src/stt/`, `src/transcriber.py`)**
   Supports `Qwen3-ASR`, `SenseVoiceSmall`, `FunASR`, and `mlx-whisper`. Uses moving-average energy chunking, character-level forced alignment (`Qwen3-ForcedAligner-0.6B-8bit`), cross-boundary orphan stitching, and toneless-pinyin homophone normalization (`src/stt/name_fixer.py`).
3. **Speaker Diarization & Attribution (`src/diarizer.py`, `src/diarization_v2/`)**
   Combines `pyannote/speaker-diarization-community-1` turn clustering, Alibaba 3D-Speaker `CAM++` 192-D voiceprint embeddings, `librosa.yin` pitch ($F_0$) gender estimation, and grammatical sentence-continuity locking so speakers do not switch mid-sentence. Optional `--diarize` adds InsightFace tracking and lip-motion active speaker detection.
4. **Context-Aware Translation (`src/translator.py`, `src/llm_backends.py`)**
   Runs a two-pass workflow: first building a show dossier (`*_brief.json`) of character names, terminology, and ASR phonetic repairs, then translating batches with surrounding context. Supports DeepSeek, OpenAI, Anthropic, OpenRouter, Groq, or local `.gguf` models via `llama.cpp`.
5. **Dubbing & Muxing (`src/dubber.py`, `src/media_engine.py`)**
   Groups consecutive same-speaker clauses into complete sentences, synthesizes speech via local `KittenTTS` (24 kHz ONNX) or cloud `edge-tts`, time-stretches clips to fit slot headroom, mixes with the Demucs accompaniment stem, and muxes audio and switchable subtitles via `ffmpeg`.

---

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.10+
- `ffmpeg` (`brew install ffmpeg`)

### Installation

```bash
git clone https://github.com/AnsgenixD/TranslateThisShow-.git
cd TranslateThisShow-
pip install -r requirements.txt
```

### Optional Modules

Optional components are loaded dynamically when installed:

- **Qwen3-ASR, SenseVoice & Forced Alignment**:
  ```bash
  pip install mlx-audio
  ```
- **On-Screen Subtitle OCR**:
  ```bash
  pip install mlx-vlm opencv-python Pillow
  ```
- **Audio-Visual Speaker Diarization (`--diarize`)**:
  ```bash
  pip install pyannote.audio insightface opencv-python
  export HF_TOKEN="your_huggingface_token"
  ```
- **On-Device KittenTTS Dubbing**:
  ```bash
  pip install kittentts phonemizer espeakng-loader
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

### Local Offline Translation (`llama.cpp`)

Run translation locally on Apple GPU using any `.gguf` model:

```bash
python main.py --video videos/my_video.mp4 --mode subbed --local --model-path models/model.gguf
```

---

## CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--video` | `videos/sample.mp4` | Path to the input video file. |
| `--mode` | `subbed` | `subbed` embeds a switchable English subtitle track; `dubbed` generates a multi-voice dubbed audio track plus subtitles. |
| `--output` | `output` | Root directory for generated stems, SRTs, and output videos. |
| `--stt-engine` | `qwen3` | Speech recognition backend: `qwen3`, `whisper`, `funasr`, `sensevoice`. |
| `--whisper-model` | `mlx-community/whisper-large-v3-mlx` | Model repo used when `--stt-engine whisper` is selected. |
| `--tts-engine` | `kittentts` | Dubbing synthesis backend: `kittentts` (local ONNX) or `edge-tts` (cloud). |
| `--diarize` | `False` | Enable face tracking and active speaker detection in addition to acoustic clustering. |
| `--ocr-assist` | `False` | Scan video frames for burned-in Chinese subtitles to seed the glossary. |
| `--provider` | `deepseek` | Translation backend: `deepseek`, `openai`, `anthropic`, `openrouter`, `groq`, `llama_cpp`. |
| `--local` | `False` | Shortcut for `--provider llama_cpp`. |
| `--model` | `""` | Override the default model name for the selected provider. |
| `--base-url` | `""` | Custom API endpoint (for `custom_openai` / `custom_anthropic`). |
| `--model-path` | `""` | Path to a local `.gguf` file (auto-detected in `models/` if omitted with `--local`). |
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
