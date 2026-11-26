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
