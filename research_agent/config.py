"""
Central configuration for the Research Agent.
Values can be overridden via environment variables or a .env file.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    openai_temperature: float = 0.3
    openai_max_tokens: int = 2048

    # ── Search APIs ───────────────────────────────────────────────────────────
    semantic_scholar_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    )
    arxiv_max_results: int = 10
    semantic_scholar_max_results: int = 10

    # ── Summarisation ─────────────────────────────────────────────────────────
    summary_max_length: int = 300        # words
    summary_style: str = "academic"      # academic | plain

    # ── Reports ───────────────────────────────────────────────────────────────
    report_output_dir: str = field(
        default_factory=lambda: os.getenv("REPORT_OUTPUT_DIR", "reports")
    )
    citation_style: str = "apa"          # apa | mla | ieee | chicago

    # ── Hypothesis ────────────────────────────────────────────────────────────
    hypothesis_count: int = 3

    # ── Misc ──────────────────────────────────────────────────────────────────
    request_timeout: int = 30            # seconds


# Shared singleton
config = Config()
