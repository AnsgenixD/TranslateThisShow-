# TranslateThisShow! 🎬

> **Broadcast-Grade AI Subtitling, Audio-Visual Diarization & Neural Dubbing Studio for Variety & Reality TV**  
> *Optimized for Apple Silicon Metal GPU & Unified Memory Architecture.*

---

## 🌟 Overview

**TranslateThisShow!** is an automated multimedia pipeline designed to handle the toughest challenges in long-form Chinese reality TV and variety shows: rapid multi-speaker banter, overlapping crosstalk, youth slang, Chinese puns (谐音梗), and high-tempo dialogue.

### Core Capabilities:
- **Zero-Swap Neural Stem Separation**: Isolates clean vocal dialogue from rich background music (BGM) and studio laughter using `demucs-mlx`.
- **Frontier Speech-to-Text (STT V2)**: High-speed Chinese speech recognition supporting Apple Silicon native `mlx-whisper`, `Qwen3-ASR`, `SenseVoice`, and `FunASR`.
- **Autonomous Entity Discovery & Phonetic Name-Fixer**: Discovers cast names and specialized show terminology via LLM and corrects phonetic homophones using toneless-pinyin matching.
- **Optional Millisecond Character Forced-Alignment**: Snaps subtitle and dubbing cues to exact acoustic speech onsets and offsets.
- **Optional On-Screen Video OCR**: Extracts on-screen Chinese broadcast subtitles directly from video frames to provide uncompromised ground truth.
- **Multimodal Audio-Visual Diarization V2**: Fuses Pyannote acoustic turns, InsightFace ArcFace tracking, geometric lip-motion Active Speaker Detection (ASD), and 3D-Speaker CAM++ 192-D voiceprints.
- **Multi-Voice Neural Dubbing**: Generates personality-matched English neural speech via Microsoft Edge-TTS with dynamic sentence-tempo stretching and steady studio BGM mixing (zero volume pumping).

---

## 🏗️ Architecture & Codebase Map

```text
TranslateThisShow/
├── main.py                  # Master entry point CLI (subtitling, dubbing, diarization)
├── requirements.txt         # Core dependencies
├── src/
│   ├── media_engine.py      # FFmpeg demuxing, normalization & subtitle burning
│   ├── audio_separator.py   # Demucs-MLX vocal / BGM separation
│   ├── transcriber.py       # Speech transcription coordinator
│   ├── translator.py        # Two-pass context-aware dialogue translation
│   ├── isometric_translator.py # Syllable/length-constrained translation
│   ├── llm_backends.py      # Universal LLM backend (DeepSeek, OpenAI, Groq, Ollama, GGUF)
│   ├── dubber.py            # Multi-speaker neural dubbing & studio mixer
│   ├── diarizer.py          # Multimodal diarization master orchestrator
│   ├── srt_io.py            # High-precision SRT parser and formatter
│   │
│   ├── stt/                 # Broadcast Speech-to-Text Engine (V2)
│   │   ├── engines.py       # Unified MLX adapters (Qwen3, Whisper, SenseVoice, FunASR)
│   │   ├── aligner.py       # Character forced aligner & acoustic boundary snapping
│   │   ├── chunker.py       # Diarization-aware turn chunker
│   │   ├── context.py       # Autonomous show dossier & dynamic entity manager
│   │   ├── name_fixer.py    # Toneless pinyin homophone corrector
│   │   ├── segmenter.py     # Broadcast clause segmenter (≤18 chars/cue)
│   │   ├── video_ocr.py     # Frame transition detector & on-screen VLM OCR
│   │   └── pipeline.py      # BroadcastSTTPipeline end-to-end coordinator
│   │
│   └── diarization_v2/      # Audio-Visual Speaker Diarization
│       ├── phase1_pyannote.py            # Pyannote community-1 acoustic clustering
│       ├── phase3_face_tracker_asd.py    # InsightFace ArcFace + geometric lip ASD
│       ├── phase4_hungarian_cooccurrence.py # Bipartite Hungarian matching
│       ├── phase5_subtitle_mapper.py     # Subtitle to speaker mapping
│       ├── phase6_temporal_smoother.py   # Conversational HMM temporal smoothing
│       ├── campplus_embedder.py          # 3D-Speaker CAM++ voiceprint embedder
│       ├── shot_detector.py              # PySceneDetect / OpenCV shot boundary detector
│       └── local_vlm_arbiter.py          # Multimodal VLM storyboard arbiter
│
├── benchmarks/
│   └── stt_bench.py         # STT verification rig (CER, Name Recall, RTF)
├── videos/                  # Put your input .mp4 files here (.gitkeep)
└── output/                  # Generated stems, SRTs, manifests and videos (.gitkeep)
```

---

## ⚡ Quickstart

### 1. Installation

Clone the repository and install core dependencies:
```bash
git clone https://github.com/AnsgenixD/TranslateThisShow-.git
cd TranslateThisShow-
pip install -r requirements.txt
```

### 2. Optional Frontier Engines (Suggested Options)

The backend is modular and works out of the box with `mlx-whisper`. If you want frontier Chinese ASR, millisecond forced alignment, or video OCR, install the optional components below. **If omitted, the system will automatically fall back to standard models without crashing.**

#### Option A: Frontier Chinese ASR & Forced Alignment (Recommended)
```bash
pip install mlx-audio
```
*Suggested Models on Hugging Face (Apple Silicon Native):*
- `mlx-community/Qwen3-ASR-0.6B-8bit` — Ultra-fast, ~1GB VRAM, lowest Character Error Rate for conversational Chinese.
- `mlx-community/Qwen3-ASR-1.7B-8bit` — Maximum accuracy for noisy/overlapping dialogue.
- `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` — Millisecond-level character CTC alignment.
- `mlx-community/SenseVoiceSmall` — 45x realtime rich audio event detector (laughter, music, applause, emotion).

#### Option B: On-Screen Video OCR & Multimodal Storyboard Arbiter
```bash
pip install mlx-vlm opencv-python Pillow
```
*Suggested Models:*
- `mlx-community/Qwen3-VL-4B-Instruct-4bit` — Fast local visual subtitle OCR and storyboard speaker arbitration.
- `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` — Higher visual acuity.

#### Option C: Audio-Visual Speaker Diarization
```bash
pip install pyannote.audio insightface opencv-python
export HF_TOKEN="your_huggingface_token"
```

---

## 🚀 Usage

### 1. Generate Subtitled Video (`--mode subbed`)
Transcribes dialogue, discovers proper nouns, translates contextually with DeepSeek/OpenAI, and embeds switchable subtitles into the output MP4:
```bash
python main.py --video videos/my_show.mp4 --mode subbed
```

To specify the STT engine:
```bash
python main.py --video videos/my_show.mp4 --stt-engine qwen3   # Recommended for Chinese TV
python main.py --video videos/my_show.mp4 --stt-engine whisper # Standard Whisper Large-v3 Turbo
```

### 2. Full Multi-Speaker Neural English Dubbing (`--mode dubbed`)
Separates speech from BGM via Demucs, generates translated dialogue, synthesizes personality-matched Microsoft Neural speech with dynamic tempo stretching, blends studio BGM, and muxes the dubbed MP4:
```bash
python main.py --video videos/my_show.mp4 --mode dubbed
```

### 3. Diarization-Aware Attribution (`--diarize`)
Track actors visually and acoustically so speech lines are mapped to specific cast members:
```bash
python main.py --video videos/my_show.mp4 --mode dubbed --diarize
```

### 4. Translation AI Options
You can use cloud APIs or completely offline local LLMs via `llama.cpp` on Apple Silicon GPU:
```bash
# DeepSeek API (Default)
export DEEPSEEK_API_KEY="your_key"
python main.py --video videos/my_show.mp4 --provider deepseek

# OpenAI / OpenRouter / Groq
export OPENAI_API_KEY="your_key"
python main.py --video videos/my_show.mp4 --provider openai --model gpt-4o

# 100% Offline Local Apple Silicon (Metal GPU via GGUF)
python main.py --video videos/my_show.mp4 --local --model-path models/Qwen3-8B-Q4_K_M.gguf
```

---

## 📊 Benchmark Verification

Run the automated verification suite to evaluate CER, entity recognition recall, and real-time factor (RTF):
```bash
python -m benchmarks.stt_bench --audio output/sample_vocals.wav
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
