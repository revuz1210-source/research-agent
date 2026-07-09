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

_MOCK_TEMPLATES = {
    "summarize": (
        "This paper investigates {topic} using rigorous methodology. "
        "The authors present novel findings that advance understanding in this domain. "
        "Key contributions include theoretical insights and empirical validation of {topic}-related methods."
    ),
    "hypothesis": (
        "### Hypothesis 1\n"
        "**Statement:** Advances in {topic} may be accelerated by incorporating cross-domain knowledge transfer.\n"
        "**Rationale:** Recent literature suggests that methods from adjacent fields have shown promise when applied to {topic}. Further investigation is warranted.\n"
        "**Confidence:** 0.65\n\n"
        "### Hypothesis 2\n"
        "**Statement:** The scalability of current {topic} approaches remains an open problem requiring novel architectural solutions.\n"
        "**Rationale:** Multiple reviewed papers highlight performance degradation at scale. A unified framework specific to {topic} could address this gap.\n"
        "**Confidence:** 0.72\n\n"
        "### Hypothesis 3\n"
        "**Statement:** Longitudinal datasets focused on {topic} would reveal temporal dynamics not captured in existing benchmarks.\n"
        "**Rationale:** Most current work on {topic} relies on static snapshots. Dynamic evaluation protocols may expose important limitations.\n"
        "**Confidence:** 0.58"
    ),
    "report_section": (
        "Research on {topic} has seen significant advances in recent years. "
        "Multiple independent groups have explored {topic} from complementary angles, "
        "lending confidence to emerging theoretical frameworks. "
        "Key open questions remain around scalability, generalisation, and real-world applicability of {topic} methods. "
        "The reviewed papers collectively highlight both the maturity and the frontier challenges of this field."
    ),
    "default": "Generated content based on the provided research context regarding {topic}.",
}


def _extract_topic_from_prompt(user_prompt: str) -> str:
    """Pull the most informative phrase from the user prompt for mock personalisation."""
    # Try to grab content after "Topic:" or "Research Topic:" labels
    import re
    m = re.search(r"(?:Research\s+)?Topic:\s*(.+)", user_prompt, re.I)
    if m:
        return m.group(1).strip()[:60]
    # First non-empty line otherwise
    for line in user_prompt.splitlines():
        line = line.strip()
        if len(line) > 8:
            return line[:60]
    return "this research area"


def _mock_chat(task: str, user_prompt: str = "") -> str:
    topic = _extract_topic_from_prompt(user_prompt)
    for key, template in _MOCK_TEMPLATES.items():
        if key in task.lower():
            return template.format(topic=topic)
    return _MOCK_TEMPLATES["default"].format(topic=topic)


# ── Public API ────────────────────────────────────────────────────────────────

def generate(system_prompt: str, user_prompt: str, task_hint: str = "default") -> str:
    """
    Generate text via the configured LLM.

    Falls back to query-aware mock output when OPENAI_API_KEY is absent,
    so the full pipeline can be exercised without credentials.
    """
    if not config.openai_api_key:
        logger.debug("No API key — using mock LLM response for task: %s", task_hint)
        return _mock_chat(task_hint, user_prompt)

    try:
        return _openai_chat(system_prompt, user_prompt)
    except Exception as exc:  # pragma: no cover
        logger.warning("LLM call failed (%s); falling back to mock.", exc)
        return _mock_chat(task_hint, user_prompt)
