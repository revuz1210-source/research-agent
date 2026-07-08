"""
LLM client — thin wrapper around OpenAI's chat completion API.
Falls back to a mock when no API key is configured, enabling offline use.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from research_agent.config import config

logger = logging.getLogger(__name__)


# ── OpenAI helper ─────────────────────────────────────────────────────────────

def _openai_chat(system_prompt: str, user_prompt: str) -> str:
    """Send a chat completion request and return the assistant text."""
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package is required. Install it with: pip install openai"
        ) from exc

    client = openai.OpenAI(api_key=config.openai_api_key)
    response = client.chat.completions.create(
        model=config.openai_model,
        temperature=config.openai_temperature,
        max_tokens=config.openai_max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Mock (offline) helper ─────────────────────────────────────────────────────

_MOCK_RESPONSES = {
    "summarize": (
        "This paper investigates the research question using rigorous methodology. "
        "The authors present novel findings that advance understanding in this domain. "
        "Key contributions include theoretical insights and empirical validation."
    ),
    "hypothesis": (
        "1. The proposed mechanism may be influenced by previously unexplored contextual factors.\n"
        "2. Cross-domain transfer of findings could yield improvements in adjacent fields.\n"
        "3. Longitudinal analysis might reveal temporal dynamics not captured in prior work."
    ),
    "keywords": "machine learning; deep learning; neural networks; research methodology",
    "report_section": (
        "Based on the reviewed literature, the field has seen significant advances "
        "in recent years. Multiple independent research groups have corroborated core "
        "findings, lending confidence to emerging theoretical frameworks. "
        "Further work is needed to resolve open questions around scalability and generalization."
    ),
    "default": "Generated content based on the provided research context.",
}


def _mock_chat(task: str) -> str:
    for key in _MOCK_RESPONSES:
        if key in task.lower():
            return _MOCK_RESPONSES[key]
    return _MOCK_RESPONSES["default"]


# ── Public API ────────────────────────────────────────────────────────────────

def generate(system_prompt: str, user_prompt: str, task_hint: str = "default") -> str:
    """
    Generate text via the configured LLM.

    Falls back to deterministic mock output when OPENAI_API_KEY is absent,
    so the full pipeline can be exercised without credentials.
    """
    if not config.openai_api_key:
        logger.debug("No API key — using mock LLM response for task: %s", task_hint)
        return _mock_chat(task_hint)

    try:
        return _openai_chat(system_prompt, user_prompt)
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM call failed (%s); falling back to mock.", exc)
        return _mock_chat(task_hint)
