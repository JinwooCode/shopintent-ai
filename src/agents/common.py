"""Shared helpers for the CrewAI agents."""
from __future__ import annotations

import os
import re

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


def make_llm(cfg: dict, temperature: float | None = None):
    """LLM from config.yaml (provider/model/timeout). The API key comes from .env."""
    from crewai import LLM

    c = cfg["llm"]
    return LLM(model=f"{c['provider']}/{c['model']}",
               temperature=c.get("temperature", 0.4) if temperature is None else temperature,
               timeout=c.get("timeout_seconds", 30), max_retries=1)


def short_error(e: Exception, limit: int = 200) -> str:
    return f"{type(e).__name__}: {re.sub(r'\s+', ' ', str(e))[:limit]}"


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
FREE_NUMBERS = set(map(float, range(11)))   # small counts ("3 checks") are always allowed


def numbers_in(text: str) -> set[float]:
    return {float(n.replace(",", "")) for n in NUMBER.findall(text)}


def invented_numbers(text: str, facts: list[str]) -> list[float]:
    """Numbers in text that do not appear in the facts: the agent invented or derived them."""
    allowed = numbers_in(" ".join(facts)) | FREE_NUMBERS
    return sorted(n for n in numbers_in(text) if n not in allowed)


def invented_problem(text: str, facts: list[str]) -> list[str]:
    bad = invented_numbers(text, facts)
    if not bad:
        return []
    return [f"Numbers not found in the facts (invented or derived): {', '.join(f'{n:g}' for n in bad[:8])}. "
            f"Use numbers exactly as written in the facts"]


def sha_of(obj) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
