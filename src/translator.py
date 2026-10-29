import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

# ==============================================================================
# Reality TV Subtitle System Prompt
# ==============================================================================
SYSTEM_PROMPT = """You are a professional subtitle localization expert specializing in Chinese reality TV and variety shows.
Your job is to translate spoken Chinese dialogue segments into natural, witty, contemporary English subtitles.

Rules:
1. Celebrity Names: Recognize Chinese celebrity and cast names (e.g., 王子奇 -> Wang Ziqi, 徐志胜 -> Xu Zhisheng, 陈铭 -> Chen Ming, 贺峻霖 -> He Junlin). NEVER translate personal names literally (e.g. NEVER translate 王子奇 as 'Prince Qi').
2. Internet Slang & Humor: Adapt Chinese workplace banter, idioms, and variety show humor naturally (e.g., '打工人嘴替' -> 'the voice of everyday workers', '牛马天花板' -> 'the ultimate corporate grinder', '加油团' -> 'Cheering Panel').
3. Pacing: Keep translations concise and natural so viewers can easily read them in the given time window.
4. Output Format: You must return ONLY a valid JSON array of objects with keys 'id' and 'en'. No conversational filler."""

USER_INSTRUCTION_TEMPLATE = """Translate the following subtitle segments from Chinese to English:
{segments_json}

Return strictly a JSON array:
[
  {{"id": 0, "en": "translated text"}},
  ...
]"""


class UniversalTranslator:
    """
    Universal Subtitle Translator supporting:
    - OpenRouter (openrouter/free or custom models)
    - DeepSeek API (deepseek-chat)
    - Anthropic Claude Format (/v1/messages)
    - Generic OpenAI-Compatible Endpoints (Ollama, Groq, vLLM, Local AI)
    - Local MLX LLM (Qwen2.5 / Gemma 2 on Apple Silicon GPU)
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        
        # Load API keys based on provider
        if self.provider == "deepseek":
            self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or ""
            self.base_url = base_url or "https://api.deepseek.com/chat/completions"
            self.model = model or "deepseek-chat"
        elif self.provider == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
            self.base_url = base_url or "https://api.anthropic.com/v1/messages"
            self.model = model or "claude-3-5-sonnet-20241022"
        elif self.provider == "mlx_local":
            self.api_key = ""
            self.base_url = ""
            # Default to compact native Chinese bilingual model
            self.model = model or "mlx-community/Qwen2.5-7B-Instruct-4bit"
        else: # Default: openrouter
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
            self.base_url = base_url or "https://openrouter.ai/api/v1/chat/completions"
            self.model = model or "openrouter/auto"

    def _call_openai_compatible(self, segments_payload: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/ansgenix/TranslationProject"
            headers["X-Title"] = "AI Video Dubbing Studio"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_INSTRUCTION_TEMPLATE.format(segments_json=segments_payload)}
            ],
            "temperature": 0.3
        }

        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]

    def _call_anthropic(self, segments_payload: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": USER_INSTRUCTION_TEMPLATE.format(segments_json=segments_payload)}
            ],
            "temperature": 0.3
        }

        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["content"][0]["text"]

    def _call_mlx_local(self, segments_payload: str) -> str:
        """
        Runs a local LLM (like Qwen2.5) entirely on your Mac's GPU via mlx-lm!
        """
        from mlx_lm import load, generate
        print(f"Loading local MLX LLM: {self.model} on Apple GPU...")
        model, tokenizer = load(self.model)
        
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{USER_INSTRUCTION_TEMPLATE.format(segments_json=segments_payload)}<|im_end|>\n<|im_start|>assistant\n"
        
        reply = generate(model, tokenizer, prompt=prompt, max_tokens=1024, verbose=False)
        return reply

    def translate_batch(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Translates a batch of subtitle segments with in-context slang understanding.
        """
        payload_items = [{"id": i, "zh": seg["text"].strip()} for i, seg in enumerate(segments)]
        payload_str = json.dumps(payload_items, ensure_ascii=False)

        print(f"🌐 Querying [{self.provider.upper()}] with model '{self.model}'...")
        
        if self.provider == "anthropic":
            raw_reply = self._call_anthropic(payload_str)
        elif self.provider == "mlx_local":
            raw_reply = self._call_mlx_local(payload_str)
        else:
            raw_reply = self._call_openai_compatible(payload_str)

        # Parse JSON response
        try:
            cleaned_reply = raw_reply.strip()
            if "```" in cleaned_reply:
                parts = cleaned_reply.split("```")
                cleaned_reply = parts[1]
                if cleaned_reply.startswith("json"):
                    cleaned_reply = cleaned_reply[4:]
                cleaned_reply = cleaned_reply.strip()
            
            translations = json.loads(cleaned_reply)
            lookup = {item["id"]: item["en"] for item in translations}
            
            for i, seg in enumerate(segments):
                seg["translated_text"] = lookup.get(i, seg.get("translated_text", seg["text"]))
        except Exception as e:
            print(f"⚠️ Warning: Could not parse LLM JSON ({e}). Raw reply: {raw_reply[:140]}...")
            for seg in segments:
                if "translated_text" not in seg:
                    seg["translated_text"] = seg["text"]

        return segments


def translate_dialogue_segments(
    segments: List[Dict[str, Any]],
    provider: str = "openrouter",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Main entry point for translation:
    - provider="openrouter" (free/paid cloud models)
    - provider="deepseek" (DeepSeek V3 / R1)
    - provider="anthropic" (Claude 3.5 Sonnet)
    - provider="mlx_local" (Local Qwen2.5 / Gemma on Apple GPU via mlx-lm)
    - provider="openai_compatible" (Ollama, Groq, vLLM)
    """
    translator = UniversalTranslator(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url
    )
    return translator.translate_batch(segments)
