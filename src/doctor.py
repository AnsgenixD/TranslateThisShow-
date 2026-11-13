#!/usr/bin/env python3
"""
doctor.py - Interactive Setup, Model Installer, Secrets Configurator & Runtime Debugger.

Built for macOS Apple Silicon (MLX). Can be run:
  1. Standalone via `python setup_and_debug.py` or `python main.py --setup`
  2. Automatically when `main.py` catches a missing package, model weight, or API key error.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import traceback
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"


# ==============================================================================
# 1. Hardware Detection (macOS Apple Silicon Profiler)
# ==============================================================================
@dataclass
class MacSpecs:
    os_name: str
    macos_version: str
    arch: str
    chip_name: str
    ram_gb: int
    cpu_cores: int
    free_disk_gb: float
    metal_available: bool
    is_apple_silicon: bool


def _sysctl(key: str) -> str:
    try:
        out = subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return ""


def detect_mac_specs() -> MacSpecs:
    os_name = platform.system()
    arch = platform.machine()
    is_apple_silicon = (os_name == "Darwin" and arch == "arm64")

    macos_version = ""
    if os_name == "Darwin":
        try:
            macos_version = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
        except Exception:
            macos_version = platform.mac_ver()[0]

    chip_name = _sysctl("machdep.cpu.brand_string") or platform.processor() or arch
    mem_bytes_str = _sysctl("hw.memsize")
    if mem_bytes_str.isdigit():
        ram_gb = int(round(int(mem_bytes_str) / (1024 ** 3)))
    else:
        try:
            import psutil
            ram_gb = int(round(psutil.virtual_memory().total / (1024 ** 3)))
        except Exception:
            ram_gb = 8

    cpu_cores = os.cpu_count() or 8
    try:
        free_disk_gb = round(shutil.disk_usage(REPO_ROOT).free / (1024 ** 3), 1)
    except Exception:
        free_disk_gb = 0.0

    metal_available = False
    try:
        import mlx.core as mx
        metal_available = bool(mx.metal.is_available())
    except Exception:
        pass

    return MacSpecs(
        os_name=os_name,
        macos_version=macos_version,
        arch=arch,
        chip_name=chip_name,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        free_disk_gb=free_disk_gb,
        metal_available=metal_available,
        is_apple_silicon=is_apple_silicon,
    )


# ==============================================================================
# 2. Curated mlx-community Model Catalog (Apple Silicon Native Only)
# ==============================================================================
@dataclass(frozen=True)
class MLXModelOption:
    repo_id: str
    category: str          # "llm" | "stt" | "aligner" | "vlm" | "tts"
    title: str
    size_gb: float
    min_ram_gb: int
    ideal_ram_gb: int
    est_speed: str
    description: str
    required_by_default: bool = False


MLX_LLM_CATALOG: List[MLXModelOption] = [
    MLXModelOption(
        repo_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
        category="llm",
        title="Qwen 2.5 3B Instruct (4-bit)",
        size_gb=2.0,
        min_ram_gb=8,
        ideal_ram_gb=8,
        est_speed="~65-95 tok/s",
        description="Ultra-lightweight and fast. Best choice for 8 GB Unified Memory Macs.",
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        category="llm",
        title="Qwen 2.5 7B Instruct (4-bit) [Default Offline]",
        size_gb=4.5,
        min_ram_gb=8,
        ideal_ram_gb=16,
        est_speed="~35-55 tok/s",
        description="Strong balance of speed and Mandarin-to-English phrasing. Ideal for 16 GB+ Macs.",
        required_by_default=True,
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen3-8B-4bit",
        category="llm",
        title="Qwen 3 8B (4-bit)",
        size_gb=5.0,
        min_ram_gb=16,
        ideal_ram_gb=16,
        est_speed="~30-45 tok/s",
        description="High accuracy on Chinese variety-show slang, idioms, and ASR error auditing.",
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen2.5-14B-Instruct-4bit",
        category="llm",
        title="Qwen 2.5 14B Instruct (4-bit)",
        size_gb=8.5,
        min_ram_gb=16,
        ideal_ram_gb=24,
        est_speed="~18-28 tok/s",
        description="Studio-grade nuance and speaker voice consistency. Ideal for 24 GB / 32 GB+ Macs.",
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen3-14B-4bit",
        category="llm",
        title="Qwen 3 14B (4-bit)",
        size_gb=8.8,
        min_ram_gb=16,
        ideal_ram_gb=24,
        est_speed="~16-25 tok/s",
        description="Latest-generation 14B Mandarin-to-English reasoning and context fidelity.",
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen2.5-32B-Instruct-4bit",
        category="llm",
        title="Qwen 2.5 32B Instruct (4-bit)",
        size_gb=18.5,
        min_ram_gb=32,
        ideal_ram_gb=36,
        est_speed="~10-16 tok/s",
        description="Maximum offline translation quality. Built for 36 GB / 48 GB / 64 GB+ Max/Ultra Macs.",
    ),
]


MLX_PIPELINE_MODELS: List[MLXModelOption] = [
    # Speech-to-Text (STT) & Forced Alignment (mlx-community)
    MLXModelOption(
        repo_id="mlx-community/Qwen3-ASR-0.6B-8bit",
        category="stt",
        title="Qwen3-ASR 0.6B (8-bit) [Default STT]",
        size_gb=0.7,
        min_ram_gb=8,
        ideal_ram_gb=8,
        est_speed="~35x realtime",
        description="Primary Mandarin broadcast speech-to-text engine (--stt-engine qwen3).",
        required_by_default=True,
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
        category="aligner",
        title="Qwen3-ForcedAligner 0.6B (8-bit) [Default Aligner]",
        size_gb=0.7,
        min_ram_gb=8,
        ideal_ram_gb=8,
        est_speed="~50x realtime",
        description="Millisecond-accurate character-level CTC timestamp aligner for subtitles.",
        required_by_default=True,
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen3-ASR-1.7B-8bit",
        category="stt",
        title="Qwen3-ASR 1.7B (8-bit) [High-Accuracy STT]",
        size_gb=1.9,
        min_ram_gb=16,
        ideal_ram_gb=16,
        est_speed="~18x realtime",
        description="Larger Mandarin ASR model for heavy cross-talk and rapid dialogue.",
        required_by_default=False,
    ),
    MLXModelOption(
        repo_id="mlx-community/whisper-large-v3-mlx",
        category="stt",
        title="Whisper Large-v3 MLX [Secondary Verifier & Whisper Engine]",
        size_gb=3.0,
        min_ram_gb=8,
        ideal_ram_gb=16,
        est_speed="~12x realtime",
        description="Used by --stt-engine whisper and closed-loop ASR acoustic re-verification.",
        required_by_default=True,
    ),
    MLXModelOption(
        repo_id="mlx-community/SenseVoiceSmall",
        category="stt",
        title="SenseVoiceSmall MLX [Fast ASR & Emotion Tagging]",
        size_gb=0.5,
        min_ram_gb=8,
        ideal_ram_gb=8,
        est_speed="~45x realtime",
        description="Ultra-fast Mandarin ASR with paralinguistic event detection (--stt-engine sensevoice).",
        required_by_default=False,
    ),
    # Vision-Language Models (VLM) for On-Screen Subtitle OCR & Visual Speaker Arbitration (mlx-community)
    MLXModelOption(
        repo_id="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        category="vlm",
        title="Qwen3-VL 4B Instruct (4-bit) [Default Video OCR VLM]",
        size_gb=2.8,
        min_ram_gb=8,
        ideal_ram_gb=16,
        est_speed="~40 tok/s",
        description="Reads burned-in Chinese subtitles from video frames (--ocr-assist) & visual speaker cues.",
        required_by_default=True,
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        category="vlm",
        title="Qwen2.5-VL 3B Instruct (4-bit) [Lightweight VLM]",
        size_gb=2.1,
        min_ram_gb=8,
        ideal_ram_gb=8,
        est_speed="~50 tok/s",
        description="Compact vision-language model for 8 GB Macs running --ocr-assist.",
        required_by_default=False,
    ),
    MLXModelOption(
        repo_id="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        category="vlm",
        title="Qwen2.5-VL 7B Instruct (4-bit) [High-Detail VLM]",
        size_gb=4.8,
        min_ram_gb=16,
        ideal_ram_gb=24,
        est_speed="~25 tok/s",
        description="Higher-resolution OCR and multi-face storyboard speaker arbitration for 24 GB+ Macs.",
        required_by_default=False,
    ),
    # On-Device Neural Dubbing TTS
    MLXModelOption(
        repo_id="KittenML/kitten-tts-nano-0.1",
        category="tts",
        title="KittenTTS Nano 0.1 [Default Multi-Voice English Dubbing]",
        size_gb=0.08,
        min_ram_gb=8,
        ideal_ram_gb=8,
        est_speed="~20x realtime",
        description="On-device multi-speaker English neural TTS engine (--tts-engine kittentts).",
        required_by_default=True,
    ),
]


def evaluate_model_fit(model: MLXModelOption, specs: MacSpecs) -> Tuple[str, str]:
    """
    Returns (tier_badge, explanation) describing how well `model` will run on `specs`.
    Accounts for macOS Unified Memory headroom (OS + audio/video buffers need ~4 GB).
    """
    ram = specs.ram_gb
    usable_ram = max(2.0, ram - 3.5)

    if not specs.is_apple_silicon:
        return ("UNSUPPORTED", "Requires macOS with Apple Silicon (M1/M2/M3/M4) for MLX Metal acceleration.")

    if model.size_gb <= usable_ram * 0.45 and ram >= model.ideal_ram_gb:
        return (
            "EXCELLENT",
            f"Uses ~{model.size_gb:.1f} GB of your {ram} GB Unified Memory ({model.est_speed}). Leaves ample headroom for STT/Demucs.",
        )
    if model.size_gb <= usable_ram * 0.75 and ram >= model.min_ram_gb:
        return (
            "GOOD",
            f"Uses ~{model.size_gb:.1f} GB of your {ram} GB Unified Memory ({model.est_speed}). Runs smoothly if heavy apps are closed.",
        )
    if model.size_gb <= usable_ram:
        return (
            "TIGHT FIT",
            f"Uses ~{model.size_gb:.1f} GB on a {ram} GB Mac. May swap to SSD during long batches; close browser tabs before running.",
        )
    return (
        "TOO LARGE",
        f"Needs ~{model.size_gb:.1f} GB weights (recommended {model.ideal_ram_gb} GB+ RAM), which exceeds safe limits on {ram} GB RAM.",
    )


# ==============================================================================
# 3. Package & System Dependency Checks
# ==============================================================================
# (import_name, pip_spec, purpose, required)
REQUIRED_PACKAGES: List[Tuple[str, str, str, bool]] = [
    ("mlx.core", "mlx>=0.22.0", "Apple Silicon MLX tensor engine", True),
    ("mlx_whisper", "mlx-whisper>=0.4.0", "MLX Whisper speech recognition", True),
    ("mlx_audio", "mlx-audio>=0.1.0", "Qwen3-ASR, Qwen3-ForcedAligner & SenseVoice on MLX", True),
    ("mlx_lm", "mlx-lm>=0.21.0", "Offline local LLM translation with mlx-community models", True),
    ("mlx_vlm", "mlx-vlm>=0.1.0", "On-screen video subtitle OCR & VLM speaker arbitration", True),
    ("demucs_mlx", "demucs-mlx>=1.6.0", "MLX HTDemucs vocal/BGM source separation", True),
    ("soundfile", "soundfile>=0.12.0", "Lossless WAV audio I/O", True),
    ("librosa", "librosa>=0.10.0", "Acoustic F0 pitch tracking & dubbing time-stretch", True),
    ("numpy", "numpy>=1.26.0", "Numerical array processing", True),
    ("scipy", "scipy>=1.11.0", "Signal processing & Hungarian speaker matching", True),
    ("torch", "torch>=2.0.0", "PyTorch runtime for Pyannote & 3D-Speaker CAM++", True),
    ("torchaudio", "torchaudio>=2.0.0", "Kaldi FBank features & audio resampling", True),
    ("kittentts", "kittentts>=0.1.3", "On-device neural English TTS dubbing engine", True),
    ("phonemizer", "phonemizer>=3.6.0", "Phoneme conversion for KittenTTS", True),
    ("espeakng_loader", "espeakng-loader>=0.2.4", "Bundled espeak-ng shared library for macOS", True),
    ("edge_tts", "edge-tts>=6.1.0", "Cloud neural English TTS fallback engine", True),
    ("pyannote.audio", "pyannote.audio", "Acoustic speaker turn diarization", True),
    ("cv2", "opencv-python", "Video frame extraction for OCR & face tracking", True),
    ("PIL", "Pillow", "Image processing for VLM OCR", True),
    ("opencc", "opencc-python-reimplemented", "Traditional-to-Simplified Chinese normalization", True),
    ("pypinyin", "pypinyin", "Mandarin homophone & phonetic name correction", True),
    ("huggingface_hub", "huggingface_hub", "Model weight downloader for mlx-community", True),
    ("httpx", "httpx>=0.25.0", "HTTP client", True),
    ("requests", "requests>=2.31.0", "HTTP client", True),
    ("pydantic", "pydantic>=2.0.0", "Structured schema validation", True),
    ("psutil", "psutil>=5.9.0", "System memory & process monitor", True),
    ("insightface", "insightface", "Optional visual face clustering (--diarize)", False),
    ("onnxruntime", "onnxruntime", "ONNX runtime for KittenTTS & InsightFace", True),
]

CAMPPLUS_WEIGHTS_PATH = (
    Path.home()
    / ".cache/modelscope/models/damo--speech_campplus_sv_zh-cn_16k-common/snapshots/master/campplus_cn_common.bin"
)
CAMPPLUS_MODELSCOPE_URL = (
    "https://www.modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo"
    "?Revision=master&FilePath=campplus_cn_common.bin"
)
SPEAKER3D_DIR = REPO_ROOT / "3D-Speaker"


def is_hf_model_cached(repo_id: str) -> bool:
    """Checks whether a Hugging Face model repo is already downloaded locally."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    folder_name = "models--" + repo_id.replace("/", "--")
    snaps = cache_root / folder_name / "snapshots"
    if not snaps.exists():
        return False
    for snap in snaps.iterdir():
        if snap.is_dir() and any(snap.iterdir()):
            return True
    return False


def download_hf_model(repo_id: str) -> bool:
    """Downloads a repository from Hugging Face Hub with progress output."""
    try:
        from huggingface_hub import snapshot_download
        print(f"   -> Downloading '{repo_id}' from Hugging Face Hub...")
        snapshot_download(repo_id=repo_id)
        print(f"   [OK] Installed '{repo_id}'.")
        return True
    except Exception as e:
        print(f"   [ERROR] Failed to download '{repo_id}': {e}")
        return False


def ensure_campplus_installed(auto_fix: bool = True) -> bool:
    """Verifies (and optionally installs) 3D-Speaker source and CAM++ 192-D speaker weights."""
    ok = True

    # 1. Check 3D-Speaker repository for speakerlab.models.campplus.DTDNN
    dtdnn_file = SPEAKER3D_DIR / "speakerlab" / "models" / "campplus" / "DTDNN.py"
    if not dtdnn_file.exists():
        if not auto_fix:
            print(f"   [MISSING] 3D-Speaker module not found at {SPEAKER3D_DIR}")
            ok = False
        else:
            print(f"   -> Cloning Alibaba 3D-Speaker into {SPEAKER3D_DIR} ...")
            try:
                subprocess.check_call(
                    ["git", "clone", "--depth", "1", "https://github.com/modelscope/3D-Speaker.git", str(SPEAKER3D_DIR)]
                )
                print("   [OK] Cloned 3D-Speaker.")
            except Exception as e:
                print(f"   [ERROR] Could not clone 3D-Speaker: {e}")
                ok = False

    # 2. Check campplus_cn_common.bin weights
    if not CAMPPLUS_WEIGHTS_PATH.exists() or CAMPPLUS_WEIGHTS_PATH.stat().st_size < 1_000_000:
        if not auto_fix:
            print(f"   [MISSING] CAM++ speaker embedding weights not found at {CAMPPLUS_WEIGHTS_PATH}")
            ok = False
        else:
            print("   -> Downloading Alibaba 3D-Speaker CAM++ weights (campplus_cn_common.bin)...")
            try:
                CAMPPLUS_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(CAMPPLUS_MODELSCOPE_URL, str(CAMPPLUS_WEIGHTS_PATH))
                print(f"   [OK] Downloaded CAM++ weights ({CAMPPLUS_WEIGHTS_PATH.stat().st_size / (1024*1024):.1f} MB).")
            except Exception as e:
                print(f"   [ERROR] Failed to download CAM++ weights: {e}")
                ok = False
    return ok


def verify_espeak_binding() -> Tuple[bool, str]:
    """Verifies that phonemizer + espeakng_loader can initialize for KittenTTS on macOS."""
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        lib_path = espeakng_loader.get_library_path()
        data_path = espeakng_loader.get_data_path()
        EspeakWrapper.set_library(str(lib_path))
        if hasattr(EspeakWrapper, "set_data_path"):
            EspeakWrapper.set_data_path(str(data_path))
        else:
            os.environ["ESPEAK_DATA_PATH"] = str(data_path)
        return True, f"espeak-ng bound ({Path(lib_path).name})"
    except Exception as e:
        return False, str(e)


def verify_project_modules() -> List[Tuple[str, bool, str]]:
    """Attempts to import all core project modules to detect syntax or code-level bugs."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    modules = [
        "src.media_engine",
        "src.audio_separator",
        "src.srt_io",
        "src.llm_backends",
        "src.translator",
        "src.transcriber",
        "src.stt.pipeline",
        "src.stt.video_ocr",
        "src.diarizer",
        "src.dubber",
    ]
    results = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            results.append((mod, True, "OK"))
        except Exception as e:
            results.append((mod, False, f"{type(e).__name__}: {e}"))
    return results


# ==============================================================================
# 4. .env Secrets & Configuration Manager
# ==============================================================================
def read_env_file(path: Path = ENV_FILE) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.is_file():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip().strip("'").strip('"')
    return data


def write_env_file(updates: Dict[str, str], path: Path = ENV_FILE) -> None:
    current = read_env_file(path)
    for k, v in updates.items():
        if v is not None:
            current[k] = v
            if v:
                os.environ[k] = v

    lines = [
        "# TranslateThisShow Environment & Secrets Configuration",
        "# Generated by setup_and_debug.py (gitignored)",
        "",
    ]
    for k, v in current.items():
        lines.append(f'{k}="{v}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mask_secret(val: str) -> str:
    if not val:
        return "not set"
    if len(val) <= 8:
        return "****"
    return f"{val[:4]}...{val[-4:]}"


# ==============================================================================
# 5. Error Diagnosis (when triggered automatically by an exception in main.py)
# ==============================================================================
def diagnose_runtime_exception(err: BaseException) -> Dict[str, str]:
    """
    Analyzes an exception raised during pipeline execution and identifies the root cause
    and recommended automated repair.
    """
    msg = str(err)
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    diagnosis = {
        "type": type(err).__name__,
        "summary": msg,
        "category": "unknown",
        "remedy": "Run full environment check, package installer, and model verification.",
    }

    if isinstance(err, (ImportError, ModuleNotFoundError)):
        diagnosis["category"] = "missing_package"
        diagnosis["remedy"] = "Install the missing Python dependency into the active environment."
    elif "No API key found" in msg or "API_KEY" in msg or "HTTP 401" in msg or "HTTP 402" in msg:
        diagnosis["category"] = "missing_api_key"
        diagnosis["remedy"] = (
            "Configure your DeepSeek / OpenAI-compatible API key in .env, "
            "or switch to offline Apple Silicon translation (mlx-community)."
        )
    elif "needs a base_url" in msg or "needs a model name" in msg or "Could not reach" in msg:
        diagnosis["category"] = "llm_connection"
        diagnosis["remedy"] = "Configure your OpenAI-compatible endpoint URL/model or switch to offline MLX translation."
    elif "campplus_cn_common.bin" in msg or "speakerlab" in msg or "3D-Speaker" in msg:
        diagnosis["category"] = "missing_campplus"
        diagnosis["remedy"] = "Download Alibaba 3D-Speaker CAM++ weights and speakerlab modules."
    elif "espeak" in msg.lower():
        diagnosis["category"] = "espeak_missing"
        diagnosis["remedy"] = "Install espeakng-loader and bind the bundled espeak-ng library for KittenTTS."
    elif "ffmpeg" in msg.lower() or "ffprobe" in msg.lower():
        diagnosis["category"] = "missing_ffmpeg"
        diagnosis["remedy"] = "Install ffmpeg via Homebrew (`brew install ffmpeg`)."
    elif "huggingface" in tb.lower() or "mlx_audio" in tb.lower() or "mlx_vlm" in tb.lower() or "mlx_lm" in tb.lower():
        diagnosis["category"] = "missing_model"
        diagnosis["remedy"] = "Download the required mlx-community STT / Aligner / VLM / LLM weights."
    elif isinstance(err, FileNotFoundError):
        diagnosis["category"] = "missing_file"
        diagnosis["remedy"] = "Check the input file path or download missing model weights."

    return diagnosis


# ==============================================================================
# 6. Interactive CLI Flows
# ==============================================================================
def print_hardware_report(specs: MacSpecs) -> None:
    print("\n" + "=" * 72)
    print("SYSTEM & HARDWARE PROFILE (macOS Apple Silicon Inspector)")
    print("=" * 72)
    print(f"  OS / Arch        : {specs.os_name} {specs.macos_version} ({specs.arch})")
    print(f"  Processor        : {specs.chip_name} ({specs.cpu_cores} cores)")
    print(f"  Unified Memory   : {specs.ram_gb} GB RAM")
    print(f"  Free Disk Space  : {specs.free_disk_gb} GB available")
    print(f"  Apple Metal GPU  : {'ACTIVE (MLX accelerated)' if specs.metal_available else 'UNAVAILABLE'}")
    print("=" * 72)


def print_mlx_llm_recommendations(specs: MacSpecs) -> None:
    print("\n" + "-" * 72)
    print(f"OFFLINE TRANSLATION MODELS (mlx-community) FOR YOUR {specs.ram_gb} GB MAC")
    print("-" * 72)
    for idx, m in enumerate(MLX_LLM_CATALOG, start=1):
        badge, advice = evaluate_model_fit(m, specs)
        cached_tag = "Cached" if is_hf_model_cached(m.repo_id) else "Not downloaded"
        print(f"  [{idx}] {m.title}")
        print(f"      Repo   : {m.repo_id} ({m.size_gb:.1f} GB | {cached_tag})")
        print(f"      Fit    : [{badge}] {advice}")
        print(f"      Notes  : {m.description}")
        print()


def print_pipeline_models_status(specs: MacSpecs) -> None:
    print("\n" + "-" * 72)
    print("STT, FORCED ALIGNER, VLM (OCR) & DUBBING MODELS")
    print("-" * 72)
    for idx, m in enumerate(MLX_PIPELINE_MODELS, start=1):
        badge, _ = evaluate_model_fit(m, specs)
        status = "INSTALLED" if is_hf_model_cached(m.repo_id) else "MISSING"
        req = "Required" if m.required_by_default else "Optional"
        print(
            f"  [{idx}] [{status:9s}] {m.repo_id} ({m.size_gb:.2f} GB | {req} | Fit: {badge})\n"
            f"                  {m.description}"
        )
    camp_status = "INSTALLED" if (CAMPPLUS_WEIGHTS_PATH.exists() and (SPEAKER3D_DIR / "speakerlab").exists()) else "MISSING"
    print(
        f"  [C] [{camp_status:9s}] Alibaba 3D-Speaker CAM++ (0.03 GB | Required for Diarization)\n"
        f"                  192-D speaker verification model (campplus_cn_common.bin)"
    )


def check_and_repair_environment(auto_install_packages: bool = True) -> bool:
    """Checks ffmpeg, Python packages, espeak-ng, 3D-Speaker CAM++, and project imports."""
    print("\n[1/4] Checking System Tools, Python Packages & Code Integrity...")
    all_ok = True

    # 1. Check ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"  [OK] ffmpeg found at {ffmpeg_path}")
    else:
        print("  [MISSING] 'ffmpeg' binary not found on PATH.")
        brew = shutil.which("brew")
        if brew and auto_install_packages:
            print("  -> Installing ffmpeg via Homebrew...")
            try:
                subprocess.check_call([brew, "install", "ffmpeg"])
                print("  [OK] Installed ffmpeg.")
            except Exception as e:
                print(f"  [ERROR] Could not install ffmpeg automatically: {e}")
                all_ok = False
        else:
            print("  -> Install Homebrew from https://brew.sh and run: brew install ffmpeg")
            all_ok = False

    # 2. Check Python packages
    missing_Specs: List[Tuple[str, str, bool]] = []
    for mod_name, pip_spec, purpose, req in REQUIRED_PACKAGES:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing_Specs.append((mod_name, pip_spec, req))
            tag = "MISSING" if req else "OPTIONAL MISSING"
            print(f"  [{tag}] {mod_name} ({purpose}) -> pip install '{pip_spec}'")

    if not missing_Specs:
        print(f"  [OK] All {len(REQUIRED_PACKAGES)} Python dependencies are installed.")
    elif auto_install_packages:
        to_install = [spec for _, spec, _ in missing_Specs]
        print(f"  -> Installing {len(to_install)} missing Python package(s) via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *to_install])
            print("  [OK] Missing Python packages installed.")
        except Exception as e:
            print(f"  [ERROR] pip install failed: {e}")
            all_ok = False

    # 3. Verify espeak-ng binding for KittenTTS
    espeak_ok, espeak_msg = verify_espeak_binding()
    if espeak_ok:
        print(f"  [OK] KittenTTS phoneme backend ready: {espeak_msg}")
    else:
        print(f"  [WARN] KittenTTS espeak-ng check failed: {espeak_msg}")
        all_ok = False

    # 4. Verify 3D-Speaker & CAM++ weights
    if ensure_campplus_installed(auto_fix=auto_install_packages):
        print("  [OK] Alibaba 3D-Speaker & CAM++ 192-D speaker weights verified.")
    else:
        all_ok = False

    # 5. Verify internal code imports
    mod_results = verify_project_modules()
    failed_mods = [(m, err) for m, ok, err in mod_results if not ok]
    if not failed_mods:
        print(f"  [OK] All {len(mod_results)} internal pipeline modules imported cleanly.")
    else:
        for mod, err in failed_mods:
            print(f"  [CODE ERROR] Module '{mod}' failed to import: {err}")
        all_ok = False

    return all_ok


def interactive_secrets_and_translation_setup(specs: MacSpecs) -> None:
    """Interactive prompt for API keys, online OpenAI-compatible endpoints, or offline mlx-community LLMs."""
    env_data = read_env_file()
    existing_ds = env_data.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    existing_hf = env_data.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    existing_prov = env_data.get("DEFAULT_PROVIDER", "deepseek")
    existing_model = env_data.get("DEFAULT_MODEL", "")

    print("\n[2/4] Secrets & Translation Engine Configuration (.env)")
    print("-" * 72)
    print(f"  Current DeepSeek API Key : {_mask_secret(existing_ds)}")
    print(f"  Current HF_TOKEN         : {_mask_secret(existing_hf)}")
    print(f"  Current Default Provider : {existing_prov} {f'({existing_model})' if existing_model else ''}")
    print("-" * 72)

    updates: Dict[str, str] = {}

    # Prompt specifically for DeepSeek API key (optional) as requested
    ds_input = input("Enter DeepSeek API key (optional, press Enter to skip/keep current): ").strip()
    if ds_input:
        updates["DEEPSEEK_API_KEY"] = ds_input

    hf_input = input("Enter Hugging Face token HF_TOKEN for pyannote (optional, press Enter to skip): ").strip()
    if hf_input:
        updates["HF_TOKEN"] = hf_input

    print("\nHow would you like to run Chinese-to-English dialogue translation?")
    print("  [1] Online API (DeepSeek, OpenRouter, OpenAI, or any OpenAI-compatible endpoint)")
    print("  [2] Offline Local on Apple Silicon (mlx-community models only, 100% private & free)")
    print("  [Enter] Keep current setting")
    mode_choice = input("Select translation mode [1/2/Enter]: ").strip()

    if mode_choice == "1":
        print("\nSelect Online API Provider:")
        print("  [1] DeepSeek (https://api.deepseek.com - default model: deepseek-flash)")
        print("  [2] OpenRouter (https://openrouter.ai/api/v1 - default model: openrouter/free)")
        print("  [3] OpenAI (https://api.openai.com/v1 - e.g. gpt-4o-mini)")
        print("  [4] Any OpenAI-Compatible Endpoint (Custom Base URL + Model + API Key)")
        p_choice = input("Choose provider [1-4, default=1]: ").strip() or "1"

        if p_choice == "1":
            updates["DEFAULT_PROVIDER"] = "deepseek"
            if not updates.get("DEEPSEEK_API_KEY") and not existing_ds:
                key = input("Enter DeepSeek API key: ").strip()
                if key:
                    updates["DEEPSEEK_API_KEY"] = key
            model_in = input("Enter DeepSeek model [default: deepseek-flash]: ").strip()
            if model_in:
                updates["DEFAULT_MODEL"] = model_in
        elif p_choice == "2":
            updates["DEFAULT_PROVIDER"] = "openrouter"
            key = input("Enter OpenRouter API key (press Enter to keep existing): ").strip()
            if key:
                updates["OPENROUTER_API_KEY"] = key
            model_in = input("Enter OpenRouter model [default: openrouter/free]: ").strip()
            updates["DEFAULT_MODEL"] = model_in or "openrouter/free"
        elif p_choice == "3":
            updates["DEFAULT_PROVIDER"] = "openai"
            key = input("Enter OpenAI API key (press Enter to keep existing): ").strip()
            if key:
                updates["OPENAI_API_KEY"] = key
            model_in = input("Enter OpenAI model [default: gpt-4o-mini]: ").strip()
            updates["DEFAULT_MODEL"] = model_in or "gpt-4o-mini"
        elif p_choice == "4":
            updates["DEFAULT_PROVIDER"] = "custom_openai"
            base_url = input("Enter OpenAI-compatible Base URL (e.g. https://api.groq.com/openai/v1): ").strip()
            if base_url:
                updates["CUSTOM_OPENAI_BASE_URL"] = base_url
            key = input("Enter API Key for this endpoint (optional, press Enter if none): ").strip()
            if key:
                updates["CUSTOM_OPENAI_API_KEY"] = key
            model_in = input("Enter Model ID for this endpoint: ").strip()
            if model_in:
                updates["CUSTOM_OPENAI_MODEL"] = model_in
                updates["DEFAULT_MODEL"] = model_in

    elif mode_choice == "2":
        print_mlx_llm_recommendations(specs)
        # Recommend best default index for their RAM
        rec_idx = 1 if specs.ram_gb <= 8 else (2 if specs.ram_gb <= 16 else 4)
        choice_str = input(
            f"Select an mlx-community model to install & set as default [1-{len(MLX_LLM_CATALOG)}, recommended={rec_idx}]: "
        ).strip()
        idx = int(choice_str) if choice_str.isdigit() and 1 <= int(choice_str) <= len(MLX_LLM_CATALOG) else rec_idx
        chosen = MLX_LLM_CATALOG[idx - 1]
        updates["DEFAULT_PROVIDER"] = "mlx"
        updates["DEFAULT_MODEL"] = chosen.repo_id
        updates["MLX_LLM_MODEL"] = chosen.repo_id
        print(f"\nSelected offline model: {chosen.repo_id}")
        if not is_hf_model_cached(chosen.repo_id):
            dl = input(f"Download '{chosen.repo_id}' (~{chosen.size_gb:.1f} GB) now? [Y/n]: ").strip().lower()
            if dl in ("", "y", "yes"):
                download_hf_model(chosen.repo_id)
        else:
            print(f"  [OK] '{chosen.repo_id}' is already cached on this Mac.")

    if updates:
        write_env_file(updates)
        print(f"  [OK] Saved configuration to {ENV_FILE}")


def interactive_models_installer(specs: MacSpecs) -> None:
    """Interactive installer for STT, Forced Aligner, VLM (OCR), KittenTTS, and CAM++ weights."""
    print("\n[3/4] Pipeline Model Weights Installer (STT, Forced Aligner, VLM, TTS)")
    print_pipeline_models_status(specs)

    missing_required = [m for m in MLX_PIPELINE_MODELS if m.required_by_default and not is_hf_model_cached(m.repo_id)]
    print("\nModel Installation Options:")
    print("  [1] Install all required default models (Qwen3-ASR-0.6B, ForcedAligner, Whisper-v3, Qwen3-VL-4B, KittenTTS)")
    print("  [2] Install ALL models (including optional Qwen3-ASR-1.7B, SenseVoiceSmall, Qwen2.5-VL-3B/7B)")
    print("  [3] Pick individual models by number (e.g. 1,2,6)")
    print("  [Enter] Skip model downloads for now")

    default_prompt = "1" if missing_required else ""
    choice = input(f"Select option [1/2/3/Enter]{' (recommended: 1)' if default_prompt else ''}: ").strip()

    to_download: List[MLXModelOption] = []
    if choice == "1":
        to_download = [m for m in MLX_PIPELINE_MODELS if m.required_by_default]
    elif choice == "2":
        to_download = list(MLX_PIPELINE_MODELS)
    elif choice == "3":
        nums = input(f"Enter model numbers separated by commas (1-{len(MLX_PIPELINE_MODELS)}): ").strip()
        for part in nums.split(","):
            p = part.strip()
            if p.isdigit() and 1 <= int(p) <= len(MLX_PIPELINE_MODELS):
                to_download.append(MLX_PIPELINE_MODELS[int(p) - 1])

    for model in to_download:
        if is_hf_model_cached(model.repo_id):
            print(f"  [OK] Already cached: {model.repo_id}")
        else:
            download_hf_model(model.repo_id)

    # Always ensure CAM++ & 3D-Speaker are ready
    ensure_campplus_installed(auto_fix=True)


def run_final_smoke_test() -> bool:
    """Runs a quick smoke check on LLM config and core subsystems."""
    print("\n[4/4] Running Post-Setup Verification...")
    from src.llm_backends import load_dotenv_if_present
    load_dotenv_if_present(ENV_FILE)

    env_data = read_env_file()
    provider = env_data.get("DEFAULT_PROVIDER") or os.environ.get("DEFAULT_PROVIDER", "deepseek")
    model = env_data.get("DEFAULT_MODEL") or os.environ.get("DEFAULT_MODEL", "")

    print(f"  Active Translation Provider : {provider} {f'({model})' if model else ''}")
    if provider == "deepseek":
        has_key = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
        if has_key:
            print("  [OK] DEEPSEEK_API_KEY is configured.")
        else:
            print("  [NOTICE] DEEPSEEK_API_KEY is not set yet. Pass --local to use offline MLX translation or set the key in .env.")
    elif provider == "mlx":
        mlx_model = model or os.environ.get("MLX_LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
        cached = is_hf_model_cached(mlx_model)
        print(f"  [{'OK' if cached else 'NOTICE'}] Offline MLX LLM '{mlx_model}' ({'cached locally' if cached else 'will download on first run'}).")

    print("\nSetup & diagnostics complete. Example commands:")
    print("  # Run with your configured default translation backend:")
    print("  python main.py --video videos/your_video.mp4 --mode dubbed --diarize")
    print("  # Force offline local translation on Apple Silicon (mlx-community):")
    print("  python main.py --video videos/your_video.mp4 --mode dubbed --local")
    return True


def run_diagnostics_and_repair(
    error: Optional[BaseException] = None,
    interactive: Optional[bool] = None,
    check_only: bool = False,
) -> int:
    """
    Main entry point for both standalone `setup_and_debug.py` and automatic error recovery in `main.py`.
    """
    if interactive is None:
        interactive = sys.stdin.isatty() and not check_only

    specs = detect_mac_specs()
    print_hardware_report(specs)

    if error is not None:
        diag = diagnose_runtime_exception(error)
        print("\n" + "!" * 72)
        print(f"AUTOMATIC ERROR DEBUGGER TRIGGERED ({diag['type']})")
        print("!" * 72)
        print(f"  Failure Summary : {diag['summary']}")
        print(f"  Root Cause      : {diag['category']}")
        print(f"  Recommended Fix : {diag['remedy']}")
        print("!" * 72)

        if interactive:
            ans = input("\nWould you like the debugger to fix this issue and configure requirements now? [Y/n]: ").strip().lower()
            if ans not in ("", "y", "yes"):
                return 1

    env_ok = check_and_repair_environment(auto_install_packages=not check_only)

    if check_only:
        print_mlx_llm_recommendations(specs)
        print_pipeline_models_status(specs)
        return 0 if env_ok else 1

    if interactive:
        interactive_secrets_and_translation_setup(specs)
        interactive_models_installer(specs)
        run_final_smoke_test()

    return 0 if env_ok else 1
