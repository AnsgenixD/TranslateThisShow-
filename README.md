# TranslateThisShow

End-to-end subtitling, multimodal speaker diarization, and multi-speaker neural video dubbing pipeline for long-form video, natively accelerated on Apple Silicon (`mlx` & `Metal`).

---

## 🎬 Demo Showcase (Included Video)

The repository includes both the original Creative Commons Mandarin variety talk show clip (`videos/variety_cc_demo.mp4`) and the finalized multi-voice English dubbed video (`videos/variety_cc_demo_dubbed.mp4`).

<video src="videos/variety_cc_demo_dubbed.mp4" controls="controls" width="100%">
  Your browser does not support the video tag. You can view the video directly at <a href="videos/variety_cc_demo_dubbed.mp4">videos/variety_cc_demo_dubbed.mp4</a>.
</video>

👉 **[Watch or Download Demo Dubbed Video (`videos/variety_cc_demo_dubbed.mp4`)](videos/variety_cc_demo_dubbed.mp4)**

### Turn-by-Turn Dubbing & Diarization Result

| # | Time Window | Slot | Synth | Speaker & Voice Profile | Mandarin Dialogue (ASR + OCR) | English Dubbed Utterance |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `0.00s -> 2.56s` | 2.56s | 2.67s | `SPEAKER_00` `expr-voice-2-m` (Male Host) | 简单的，比如说是干爽的意思，去讲。 | *"Simple ones, like explaining it means 'dry'."* |
| **2** | `2.88s -> 6.80s` | 3.92s | 4.16s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | 但是我情急之下就马上要翻啊，干干燥燥粉粉嫩嫩。 | *"But I had to translate on the spot, so I said 'dry-dry, soft-soft'."* |
| **3** | `7.31s -> 9.63s` | 2.32s | 2.23s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | 干干燥燥，他们就觉得很好玩。 | *"'Dry-dry,' and they thought it was hilarious."* |
| **4** | `9.63s -> 11.71s` | 2.08s | 1.75s | `SPEAKER_00` `expr-voice-2-m` (Male Host) | 就是你在讲什么？哎， | *"Like, what are you saying? Oh,"* |
| **5** | `11.75s -> 15.47s` | 3.72s | 3.76s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | 什么叠字叠字这样，而且感觉很像绕口令一样，然后成员们就上瘾。 | *"Reduplicated words like that sound just like a tongue twister, and the members got hooked."* |
| **6** | `15.51s -> 20.07s` | 4.56s | 4.65s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | 上瘾了，然后连续两天他们就一直抓着这个来跟读学习，对，一直学。 | *"They got hooked, and for two days straight they kept repeating it to follow along and learn, right, kept studying it."* |
| **7** | `20.11s -> 24.39s` | 4.28s | 4.32s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | 啊，我叫Yunie，那我对，那我的工作是韩星的翻译。 | *"Ah, I'm Yunie, and my job is interpreting for K-pop stars."* |
| **8** | `24.71s -> 28.94s` | 4.23s | 4.06s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | 那目前资历的话，入行已经超过十年了，十年。 | *"As for my experience, I've been in the industry for over ten years now. Ten years."* |
| **9** | `28.98s -> 31.87s` | 2.89s | 2.62s | `SPEAKER_00` `expr-voice-2-m` (Male Host) | 合作过的对象可怕呢，很多呢。 | *"I've done so many duet interviews, it's scary, so many."* |
| **10** | `32.27s -> 38.03s` | 5.76s | 5.35s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | SEVENTEEN、ITZY、NCT，都是很厉害的团体。 | *"SEVENTEEN, ITZY, NCT, all amazing groups."* |
| **11** | `38.03s -> 40.83s` | 2.80s | 3.03s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | 其实还有很多。很多啦，但是想说就写不完。 | *"Actually there are many more. A lot, but I couldn't write them all."* |
| **12** | `40.83s -> 42.67s` | 1.84s | 1.77s | `SPEAKER_00` `expr-voice-2-m` (Male Host) | 对对，就写几个代表就好了。 | *"Right, just a few representative ones is fine."* |
| **13** | `42.67s -> 45.18s` | 2.51s | 2.33s | `SPEAKER_00` `expr-voice-2-m` (Male Host) | 对对对，口译的时间压力啊，怎么说？ | *"Right, the time pressure of interpreting, how to put it?"* |
| **14** | `45.39s -> 47.79s` | 2.40s | 2.36s | `SPEAKER_00` `expr-voice-2-m` (Male Host) | 翻译，你要及时。他讲完之后，你就立刻翻译。 | *"Yunie, you have to be timely. The moment they finish speaking, you translate."* |
| **15** | `47.87s -> 49.87s` | 2.00s | 3.48s | `SPEAKER_01` `expr-voice-2-f` (Female Guest Yunie) | 对，因为很多我们其实是在。 | *"Yes, because a lot of the time we're actually..."* |

---

## ⚡ Key Features

1. **Stem Separation (`src/audio_separator.py`)**:
   - Isolates speech and background music via Apple Silicon native `demucs-mlx`.
   - Prevents loud TV background tracks and sound effects from triggering speech hallucinations.
   - Preserves high-fidelity stereo backing music during neural dubbing.
2. **Frontier Chinese ASR (`src/stt/`, `src/transcriber.py`)**:
   - Supports `Qwen3-ASR` (0.6B/1.7B 8-bit), `SenseVoiceSmall`, `FunASR`, and `mlx-whisper`.
   - Millisecond-accurate character-level forced alignment via `Qwen3-ForcedAligner-0.6B-8bit`.
   - Energy valley moving-average chunking with cross-boundary orphan stitching.
   - Toneless-pinyin homophone normalization (`src/stt/name_fixer.py`) for dialectal confusion (`zh/z`, `ch/c`, `sh/s`, `ang/an`, `eng/en`).
3. **Acoustic & Multimodal Speaker Diarization (`src/diarizer.py`)**:
   - `pyannote/speaker-diarization-community-1` acoustic clustering with duration-adaptive speaker bounds.
   - **Alibaba 3D-Speaker CAM++ 192-D** deep voiceprint embeddings with cosine similarity attribution.
   - **Grammatical Sentence-Continuity Locking**: Guarantees clauses separated across commas never switch speakers mid-sentence.
   - **Conversational Sandwich Smoothing**: Eliminates isolated 1-cue backchannel speaker flipping.
   - **Fundamental Frequency ($F_0$) Gender Estimation**: Analyzes pitch via `librosa.yin` to assign female/male voice presets accurately.
4. **Context-Aware LLM Translation & Closed-Loop ASR Audit (`src/translator.py`)**:
   - Two-pass structured dossier extraction (`show_dossier.json`) extracting entity names and domain glossaries.
   - Supports DeepSeek, OpenAI, Anthropic, OpenRouter, Groq, and local `.gguf` models via `llama.cpp`.
   - In-place phonetic repair of ASR errors guided by LLM contextual briefs.
5. **Multi-Speaker Neural Dubbing Engine (`src/dubber.py`)**:
   - **On-Device KittenTTS**: Ultra-lightweight ~24 MB ONNX model executing on Apple Silicon CPU at **~9× faster than real-time** with zero cloud latency.
   - **Cloud Microsoft Neural Voices**: 14+ gender-calibrated voices via `edge-tts`.
   - Coherent utterance merging across pauses so sentences are spoken with natural, uninterrupted prosody.
   - Strict slot headroom matching via phase-vocoder time-stretching (`librosa.effects.time_stretch`) ensuring speech never spills into subsequent speaker turns.
   - Studio mixdown with original Demucs BGM stem and lossless FFmpeg remuxing (`-c:v copy -c:a aac -c:s mov_text`).

---

## 🛠️ Requirements & Installation

- **OS**: macOS with Apple Silicon (M1/M2/M3/M4)
- **Python**: 3.10+
- **System Tool**: `ffmpeg` (`brew install ffmpeg`)

### 1. Clone & Install Core Dependencies

```bash
git clone https://github.com/AnsgenixD/TranslateThisShow-.git
cd TranslateThisShow-
pip install -r requirements.txt
```

### 2. Optional Frontier Accelerators

All components are dynamically loaded with graceful fallbacks:

- **Qwen3-ASR & Forced Alignment (Recommended)**:
  ```bash
  pip install mlx-audio
  ```
- **On-Screen Subtitle OCR**:
  ```bash
  pip install mlx-vlm opencv-python Pillow
  ```
- **Audio-Visual Diarization (Pyannote & Face Tracking)**:
  ```bash
  pip install pyannote.audio insightface opencv-python
  export HF_TOKEN="your_huggingface_token"
  ```
- **On-Device KittenTTS**:
  ```bash
  pip install kittentts phonemizer espeakng-loader
  ```

---

## 🚀 CLI Usage Guide

All operations run through `main.py`. Results are automatically saved to `output/<video_stem>/`.

### 1. English Subtitling Mode (`--mode subbed`)

Extracts audio, isolates vocals, transcribes speech with MLX, translates with DeepSeek, and embeds switchable English subtitles:

```bash
export DEEPSEEK_API_KEY="your_api_key"

# Default Qwen3-ASR engine
python main.py --video videos/variety_cc_demo.mp4 --mode subbed

# Using Whisper Large-v3 or SenseVoice
python main.py --video videos/variety_cc_demo.mp4 --mode subbed --stt-engine whisper
python main.py --video videos/variety_cc_demo.mp4 --mode subbed --stt-engine sensevoice
```

### 2. Multi-Voice English Dubbing Mode (`--mode dubbed`)

Isolates dialogue and BGM, attributes speech to speaker clusters, synthesizes actor voices, and mixes a dubbed audio master:

```bash
# On-Device KittenTTS (Local CPU/ONNX, ~9x realtime, zero cloud calls):
python main.py --video videos/variety_cc_demo.mp4 --mode dubbed --tts-engine kittentts

# Microsoft Cloud Neural voices:
python main.py --video videos/variety_cc_demo.mp4 --mode dubbed --tts-engine edge-tts
```

### 3. Audio-Visual Diarization (`--diarize`)

Enables Pyannote speaker clustering, face tracking, and lip active speaker detection:

```bash
python main.py --video videos/variety_cc_demo.mp4 --mode dubbed --diarize
```

### 4. Video Subtitle OCR Assist (`--ocr-assist`)

Scans on-screen burned-in Chinese subtitles from video frames using Sobel stroke-density filtering to extract named entities and bias ASR vocabulary:

```bash
python main.py --video videos/variety_cc_demo.mp4 --mode subbed --ocr-assist
```

### 5. Local Offline Translation (`llama.cpp`)

Run translation 100% locally on Apple Silicon Metal GPU with any `.gguf` model:

```bash
python main.py --video videos/variety_cc_demo.mp4 --mode subbed --local --model-path models/my_model.gguf
```

---

## 📖 CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` | `str` | `videos/sample.mp4` | Path to the input video file. |
| `--mode` | `choice` | `subbed` | `'subbed'` embeds switchable English subtitles (`.mp4`).<br>`'dubbed'` synthesizes multi-voice neural speech and mixes with BGM. |
| `--output` | `str` | `output` | Directory where intermediate stems, SRTs, and finalized videos are stored. |
| `--stt-engine` | `choice` | `qwen3` | Chinese Speech-to-Text engine: `qwen3`, `whisper`, `funasr`, `sensevoice`. |
| `--whisper-model`| `str` | `mlx-community/whisper-large-v3-mlx` | Primary Whisper model repo when `--stt-engine whisper` is selected. |
| `--tts-engine` | `choice` | `kittentts` | Neural TTS engine for dubbing:<br>• `kittentts`: Fast, lightweight (~24MB) on-device ONNX synthesis.<br>• `edge-tts`: Microsoft Azure Cloud Neural voices. |
| `--diarize` | `flag` | `False` | Run multimodal audio-visual speaker diarization and face tracking. |
| `--ocr-assist` | `flag` | `False` | Extract burned-in Chinese video subtitles to seed names and terminology. |
| `--provider` | `choice` | `deepseek` | LLM translation provider: `deepseek`, `openai`, `anthropic`, `openrouter`, `groq`, `llama_cpp`. |
| `--local` | `flag` | `False` | Use local `llama.cpp` on Apple Silicon Metal GPU (sets provider to `llama_cpp`). |
| `--model-path` | `str` | `""` | Path to local `.gguf` weights (auto-detected in `models/` if omitted with `--local`). |
| `--no-brief` | `flag` | `False` | Skip the two-pass contextual dossier analysis. |

---

## 📂 Output Artifacts Directory Structure

Running `main.py --video videos/variety_cc_demo.mp4 --mode dubbed` produces:

```text
output/variety_cc_demo/
├── variety_cc_demo_extracted.wav      # Raw demuxed 16kHz audio track
├── stems/
│   ├── vocals.wav                     # Demucs isolated speech stem
│   └── accompaniment.wav              # Demucs isolated background music & SFX stem
├── diar_v2/
│   ├── pyannote.rttm                  # Standard speaker turn RTTM
│   ├── pyannote_exclusive.rttm        # Non-overlapping speech turn RTTM
│   ├── pyannote_turns.json            # Parsed speech turns with speaker IDs
│   └── vocals_16k.wav                 # Diarization 16kHz mono audio
├── dubbing_clips/
│   ├── line_0001.wav                  # Utterance 1 synthesized clip (KittenTTS)
│   ├── line_0002.wav                  # Utterance 2 synthesized clip (KittenTTS)
│   └── ...
├── variety_cc_demo_dossier.json       # Extracted character names & domain glossary
├── variety_cc_demo_brief.json         # LLM contextual translation dossier
├── variety_cc_demo_source.srt         # Finalized Mandarin source subtitles
├── variety_cc_demo_translated.srt     # Finalized English translated subtitles
├── variety_cc_demo_dubbed.wav         # Master mixed audio track (Speech + BGM)
└── variety_cc_demo_dubbed.mp4         # Final multiplexed video with dubbing + subtitles
```

---

## ⚖️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
Mandarin variety demo video (`videos/variety_cc_demo.mp4`) used under Creative Commons Attribution licensing.
