"""
context.py - Context-Conditioned Roster & Glossary Manager for Broadcast STT.

Manages proper names, terminology, and character registries for Chinese reality TV.
Provides autonomous discovery and dynamic hotword / prompt generation for ASR models
(Qwen3-ASR, Fun-ASR-Nano, MLX Whisper).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_DOSSIER_PATH = "output/show_dossier.json"
DEFAULT_ROSTER_PATH = "output/face_roster/confirmed_actor_roster.json"


class STTContextManager:
    """
    Maintains domain knowledge, names, and terminology to bias ASR decoding
    and drive post-processing phonetic correction.
    """

    def __init__(
        self,
        dossier_path: str = DEFAULT_DOSSIER_PATH,
        roster_path: str = DEFAULT_ROSTER_PATH,
        extra_hotwords: Optional[List[str]] = None,
    ):
        self.dossier_path = Path(dossier_path)
        self.roster_path = Path(roster_path)
        self.character_registry: List[Dict[str, Any]] = []
        self.domain_glossary: Dict[str, str] = {}
        self.show_title: Dict[str, str] = {}
        self.synopsis: str = ""
        self.extra_hotwords: List[str] = extra_hotwords or []

        self.load()

    def load(self) -> None:
        """Loads domain knowledge from dossier and roster files if available."""
        if self.dossier_path.exists():
            try:
                data = json.loads(self.dossier_path.read_text(encoding="utf-8"))
                self.character_registry = data.get("character_registry", [])
                self.domain_glossary = data.get("domain_glossary", {})
                self.show_title = data.get("show_title", {})
                self.synopsis = data.get("synopsis_and_premise", "")
            except Exception as e:
                print(f"⚠️ Warning loading dossier from {self.dossier_path}: {e}")

    def discover_entities_with_llm(
        self,
        transcript_sample: List[str] | str,
        llm_config: Optional[Any] = None,
        save_dossier: bool = True,
    ) -> Dict[str, Any]:
        """
        Autonomous entity discovery:
        Feeds transcript text to an LLM (DeepSeek, OpenRouter, or local) to extract
        cast member names, mentors, guests, and show terminology without user intervention.
        """
        if isinstance(transcript_sample, list):
            sample_text = "\n".join(transcript_sample[:300])
        else:
            sample_text = str(transcript_sample)[:12000]

        if not sample_text.strip():
            return {}

        prompt_system = (
            "你是一位资深影视综艺内容分析师。请分析以下中文对话文本，提取出其中提到的人物姓名、称谓、专业术语或行话。\n"
            "严格输出JSON格式，结构如下：\n"
            "{\n"
            '  "names": [{"name_zh": "李明", "role": "嘉宾", "inferred_gender": "male"}],\n'
            '  "terms": {"耳返": "Earpiece", "机位": "Camera Angle"}\n'
            "}"
        )

        try:
            try:
                from src.llm_backends import LLMConfig, make_backend
            except ImportError:
                from CleanProject.src.llm_backends import LLMConfig, make_backend

            config = llm_config or LLMConfig(provider="deepseek")
            with make_backend(config) as llm:
                reply = llm.chat(prompt_system, sample_text)

            # Extract JSON block
            m = re.search(r"\{.*\}", reply, re.DOTALL)
            if not m:
                return {}
            data = json.loads(m.group(0))

            new_names = [n.get("name_zh") for n in data.get("names", []) if n.get("name_zh")]
            new_terms = data.get("terms", {})

            print(f"🧠 Autonomous LLM Discovery found: {len(new_names)} names, {len(new_terms)} terms.")
            self.update_from_autonomous_discovery(
                new_names=new_names,
                new_terms=new_terms,
                save_dossier=save_dossier,
            )
            return data
        except Exception as e:
            print(f"⚠️ Autonomous LLM discovery skipped ({e}). Using existing dossier.")
            return {}

    def get_all_names(self) -> List[str]:
        """Returns all registered character and person names."""
        names = []
        for c in self.character_registry:
            zh = c.get("name_zh", "").strip()
            if zh and zh not in names:
                names.append(zh)
        return names

    def get_all_terms(self) -> List[str]:
        """Returns all domain terminology and slang words."""
        terms = list(self.domain_glossary.keys())
        for hw in self.extra_hotwords:
            if hw not in terms and hw not in self.get_all_names():
                terms.append(hw)
        return terms

    def get_hotwords(self, max_count: int = 50) -> List[str]:
        """
        Returns a deduplicated list of high-priority hotwords for Qwen3-ASR and Fun-ASR-Nano.
        Priority: Cast names first, followed by key domain terms.
        """
        hotwords: List[str] = []
        # 1. Names
        for name in self.get_all_names():
            if name and name not in hotwords:
                hotwords.append(name)
        # 2. Terms
        for term in self.get_all_terms():
            if term and term not in hotwords:
                hotwords.append(term)
        return hotwords[:max_count]

    def get_system_prompt(self, max_length: int = 240) -> str:
        """
        Builds a compact system prompt / initial prompt for Whisper or Qwen3-ASR.
        Injects show title, cast members, and domain terminology.
        """
        names_str = "、".join(self.get_all_names()[:10])
        terms_str = "、".join(self.get_all_terms()[:12])
        title_zh = self.show_title.get("zh", "职场/真人秀节目")

        prompt = f"以下是《{title_zh}》对话转写。人物包括：{names_str}。专业词汇：{terms_str}。请使用规范标准简体中文及标点。"
        if len(prompt) > max_length:
            prompt = prompt[:max_length - 3] + "..."
        return prompt

    def update_from_autonomous_discovery(
        self,
        new_names: Optional[List[str]] = None,
        new_terms: Optional[Dict[str, str]] = None,
        save_dossier: bool = True,
    ) -> None:
        """
        Autonomously updates the registry and dossier when new entities are discovered.
        """
        modified = False
        existing_names = set(self.get_all_names())
        if new_names:
            for n in new_names:
                n = n.strip()
                if n and n not in existing_names:
                    self.character_registry.append({
                        "name_zh": n,
                        "name_en": "",
                        "inferred_gender": "unknown",
                        "discovered_autonomously": True,
                    })
                    existing_names.add(n)
                    modified = True

        if new_terms:
            for k, v in new_terms.items():
                k = k.strip()
                if k and k not in self.domain_glossary:
                    self.domain_glossary[k] = v.strip()
                    modified = True

        if modified and save_dossier:
            self.save_dossier()

    def save_dossier(self) -> None:
        """Saves current state to output/show_dossier.json."""
        self.dossier_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "show_title": self.show_title or {"zh": "", "en": ""},
            "genre_and_format": "",
            "synopsis_and_premise": self.synopsis,
            "character_registry": self.character_registry,
            "domain_glossary": self.domain_glossary,
        }
        with open(self.dossier_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Updated dossier saved to {self.dossier_path}")
