# AI Video Subtitling & Dubbing Studio

An end-to-end multimedia pipeline designed for Chinese reality TV and variety shows, featuring:
- **Audio Demuxing & Normalization** (FFmpeg)
- **High-Speed Transcription on Apple Silicon GPU** (`mlx-whisper`)
- **Zero-Swap Neural Audio Separation** (`demucs-mlx`)
- **Context-Aware Dialogue Translation** (Local Seq2Seq / Cloud API)
- **Dual Output Modes:** Anime-style Subtitled MP4 or Full Dubbing Audio Stems

---

## Architecture & Project Structure

```text
CleanProject/
├── videos/                  <-- Put your input .mp4 files here!
├── output/                  <-- Generated .wav, .srt, and final .mp4 files appear here
├── src/
│   ├── media_engine.py      <-- FFmpeg stream demuxing & subtitle burning
│   ├── audio_separator.py   <-- Adaptive memory-managed Demucs-MLX
│   ├── transcriber.py       <-- Metal GPU ASR with mlx-whisper
│   └── translator.py        <-- Local/API context-aware translation
├── main.py                  <-- Unified entry point (ready for future GUI)
├── requirements.txt
└── .gitignore
```

---

## Quickstart

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add a Video:**
   Copy your video into `videos/`, e.g. `videos/my_episode.mp4`.

3. **Run Subtitling Pipeline (Anime-style Subbed MP4):**
   ```bash
   python main.py --video videos/my_episode.mp4 --mode subbed
   ```

4. **Run Full Dubbing Preparation (Isolates BGM and speech stems):**
   ```bash
   python main.py --video videos/my_episode.mp4 --mode dubbed
   ```
