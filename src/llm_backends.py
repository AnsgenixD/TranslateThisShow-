"""
LLM backends for the translator.

Every backend does one job: take a system prompt and a user message, return the model's text.
The translator never needs to know which provider is behind it, so adding a provider
(or a dropdown entry in the future GUI) means adding one preset here and nothing else.

Three wire protocols cover essentially everything:
  * "openai"    - POST {base}/chat/completions, `Authorization: Bearer <key>`.
                  OpenRouter, DeepSeek, OpenAI, Groq, Together, vLLM, LM Studio, Ollama, ...
  * "anthropic" - POST {base}/v1/messages, `x-api-key: <key>`.
  * "llama_cpp" - we start a local `llama-server` process on a free port, then talk to it
                  with the same OpenAI-style protocol. It is just a local OpenAI-style server.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class LLMError(RuntimeError):
    """Any backend problem: auth, network, an unusable reply, or a server that failed to start."""


# ==============================================================================
# Provider presets (this dict is what a GUI dropdown would be built from)
# ==============================================================================
@dataclass(frozen=True)
class ProviderPreset:
    label: str
    protocol: str            # "openai" | "anthropic" | "llama_cpp"
    base_url: str
    key_env: str             # environment variable holding the API key ("" = none needed)
    default_model: str
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    note: str = ""
    # Extra JSON fields sent with every request (provider-specific switches).
    extra_body: dict = field(default_factory=dict)


PROVIDER_PRESETS = {
    "openrouter": ProviderPreset(
        "OpenRouter", "openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter/free",
        note="'openrouter/free' lets OpenRouter pick any currently-free model. Free models are rate-limited."),
    "deepseek": ProviderPreset(
        "DeepSeek", "openai", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-flash",
        # Verified against the live API: without this the model 'thinks' first, which was ~13x more
        # tokens for the same translation and sometimes used up max_tokens before answering at all.
        note="Thinking is switched off: subtitle translation does not need it.",
        extra_body={"thinking": {"type": "disabled"}}),
    "openai": ProviderPreset(
        "OpenAI", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "",
        note="Enter a model name."),
    "groq": ProviderPreset(
        "Groq", "openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "",
        note="Enter a model name."),
    "anthropic": ProviderPreset(
        "Anthropic", "anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY", "",
        auth_header="x-api-key", auth_prefix="", note="Enter a model name."),
    "llama_cpp": ProviderPreset(
        "Local (llama.cpp)", "llama_cpp", "", "", "local",
        note="Starts llama-server for you. Needs the path to a .gguf file."),
    "custom_openai": ProviderPreset(
        "Custom (OpenAI-style)", "openai", "", "", "",
        note="Any OpenAI-compatible endpoint: give base_url + model (+ api_key if it needs one)."),
    "custom_anthropic": ProviderPreset(
        "Custom (Anthropic-style)", "anthropic", "", "", "",
        auth_header="x-api-key", auth_prefix="",
        note="Any Anthropic-compatible endpoint: give base_url + model + api_key."),
}


@dataclass
class LLMConfig:
    """Everything the GUI needs to collect. All fields except `provider` are optional."""
    provider: str = "openrouter"
    model: str = ""                    # empty -> the preset's default model
    api_key: str = ""                  # empty -> read from the preset's environment variable
    base_url: str = ""                 # empty -> the preset's URL (required for custom_*)
    auth_header: str = ""              # empty -> the preset's header name
    auth_prefix: Optional[str] = None  # None -> the preset's prefix ("Bearer " for most)
    model_path: str = ""               # llama_cpp only: path to a .gguf file
    temperature: float = 0.2
    max_tokens: int = 4096             # generous: "thinking" models spend tokens before answering
    timeout_s: int = 180
    ctx_size: int = 8192               # llama_cpp only
    gpu_layers: int = 99               # llama_cpp only (99 = everything on the GPU)
    # Advanced: extra JSON fields merged into every request (provider-specific options).
    extra_body: dict = field(default_factory=dict)


# ==============================================================================
# HTTP plumbing
# ==============================================================================
def _post_json(url: str, headers: dict, payload: dict, timeout: int, retries: int = 3) -> dict:
    """POST JSON, retrying rate limits (429) and transient server errors with a growing pause."""
    body = json.dumps(payload).encode("utf-8")
    delay = 3.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise LLMError(f"HTTP {e.code} from {url}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise LLMError(f"Could not reach {url}: {e}") from None
    raise LLMError(f"Gave up on {url}")  # unreachable, keeps type checkers happy


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _clean_reply(text: Optional[str]) -> str:
    """Drops <think>...</think> reasoning blocks that some models put inside the reply."""
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    if cleaned.startswith("<think>"):  # the model ran out of tokens while still 'thinking'
        return ""
    return cleaned


def _endpoint(base_url: str, tail: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith(tail) else base + tail


# ==============================================================================
# Backends
# ==============================================================================
class _Backend:
    """Common interface. Also a context manager so `with make_backend(cfg) as llm:` cleans up."""
    context_chars = 30000  # how much transcript text we dare put in one prompt

    def chat(self, system: str, user: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class OpenAIStyleBackend(_Backend):
    def __init__(self, cfg: LLMConfig, base_url: str, model: str, headers: dict,
                 extra_body: Optional[dict] = None):
        self.cfg = cfg
        self.url = _endpoint(base_url, "/chat/completions")
        self.model = model
        self.headers = headers
        self.extra_body = extra_body or {}

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            **self.extra_body,
        }
        data = _post_json(self.url, self.headers, payload, self.cfg.timeout_s)
        if not data.get("choices"):
            raise LLMError(f"Provider returned no choices: {str(data)[:300]}")
        choice = data["choices"][0]
        text = _clean_reply(choice.get("message", {}).get("content"))
        if not text:
            raise LLMError(
                f"Empty reply (finish_reason={choice.get('finish_reason')}). "
                "Reasoning models can use up max_tokens while thinking; try a larger max_tokens.")
        return text


class AnthropicStyleBackend(_Backend):
    def __init__(self, cfg: LLMConfig, base_url: str, model: str, headers: dict):
        self.cfg = cfg
        base = base_url.rstrip("/")
        if base.endswith("/messages"):
            self.url = base
        elif base.endswith("/v1"):
            self.url = base + "/messages"
        else:
            self.url = base + "/v1/messages"
        self.model = model
        self.headers = {"anthropic-version": "2023-06-01", **headers}

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.cfg.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
        }
        data = _post_json(self.url, self.headers, payload, self.cfg.timeout_s)
        blocks = data.get("content") or []
        text = _clean_reply("".join(b.get("text", "") for b in blocks if b.get("type") == "text"))
        if not text:
            raise LLMError(f"Empty reply: {str(data)[:300]}")
        return text


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaCppBackend(OpenAIStyleBackend):
    """
    Starts `llama-server` (from llama.cpp) as a child process on a free local port, waits until
    the model is loaded, and then behaves exactly like any other OpenAI-style backend.
    The process is stopped on close() (and at interpreter exit, in case close() never runs).
    """

    def __init__(self, cfg: LLMConfig):
        binary = shutil.which("llama-server")
        if not binary:
            raise LLMError("'llama-server' not found on PATH. On macOS: `brew install llama.cpp`.")
        model_path = Path(cfg.model_path).expanduser()
        if not model_path.is_file():
            raise LLMError(f"GGUF model file not found: {model_path}")

        port = _free_port()
        self._log = tempfile.NamedTemporaryFile("w+", suffix="_llama-server.log", delete=False)
        cmd = [binary, "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port),
               "-c", str(cfg.ctx_size), "-ngl", str(cfg.gpu_layers), "--no-webui", "-a", "local"]
        print(f"🦙 Starting llama-server ({model_path.name}) on port {port} ...")
        self._proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        self._wait_until_ready(port)

        super().__init__(
            cfg, base_url=f"http://127.0.0.1:{port}/v1", model="local", headers={},
            # Qwen3-style models would otherwise 'think' for a long time before each answer.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, **cfg.extra_body})
        # Keep prompts well inside the context window (roughly 1 token per ~2 characters).
        self.context_chars = int(cfg.ctx_size * 0.35)

    def _log_tail(self, lines: int = 15) -> str:
        self._log.flush()
        return "\n".join(Path(self._log.name).read_text(errors="replace").splitlines()[-lines:])

    def _wait_until_ready(self, port: int, timeout_s: int = 300) -> None:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise LLMError("llama-server exited while starting:\n" + self._log_tail())
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        print("🦙 llama-server is ready.")
                        return
            except Exception:
                pass  # 503 while the model is still loading -> keep waiting
            time.sleep(1)
        self.close()
        raise LLMError("llama-server did not become ready in time.\n" + self._log_tail())

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def make_backend(cfg: LLMConfig) -> _Backend:
    """Turns a config (what the GUI collects) into a ready-to-use backend."""
    if cfg.provider not in PROVIDER_PRESETS:
        raise LLMError(f"Unknown provider '{cfg.provider}'. Choose one of: {', '.join(PROVIDER_PRESETS)}")
    preset = PROVIDER_PRESETS[cfg.provider]

    if preset.protocol == "llama_cpp":
        return LlamaCppBackend(cfg)

    base_url = cfg.base_url or preset.base_url
    if not base_url:
        raise LLMError(f"'{cfg.provider}' needs a base_url (the API address).")
    model = cfg.model or preset.default_model
    if not model:
        raise LLMError(f"'{cfg.provider}' needs a model name.")

    api_key = cfg.api_key or (os.environ.get(preset.key_env, "").strip() if preset.key_env else "")
    if preset.key_env and not api_key:
        raise LLMError(f"No API key found. Set the {preset.key_env} environment variable or pass api_key.")

    header = cfg.auth_header or preset.auth_header
    prefix = preset.auth_prefix if cfg.auth_prefix is None else cfg.auth_prefix
    headers = {header: prefix + api_key} if api_key else {}

    if preset.protocol == "anthropic":
        return AnthropicStyleBackend(cfg, base_url, model, headers)
    if cfg.provider == "openrouter":
        headers["X-Title"] = "Video Subtitle Translator"  # optional app name shown on OpenRouter
    return OpenAIStyleBackend(cfg, base_url, model, headers,
                              extra_body={**preset.extra_body, **cfg.extra_body})
