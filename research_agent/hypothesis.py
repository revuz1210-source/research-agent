"""
Hypothesis suggestion module.

Analyses the retrieved literature and proposes novel, testable hypotheses.
"""

from __future__ import annotations

import logging
import re
from typing import List

from research_agent.config import config
from research_agent.models import Hypothesis, Paper
from research_agent import llm_client

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_HYPOTHESIS = """\
You are an expert research scientist and critical thinker.

Your task is to generate {count} novel, testable hypotheses based on the provided \
literature summaries. Each hypothesis must:
  1. Be clearly stated as a falsifiable scientific claim.
  2. Identify a specific gap, tension, or unexplored direction in the literature.
  3. Include a concise rationale (2–3 sentences) citing the relevant papers.
  4. Be distinct from the others.

Format each hypothesis EXACTLY as follows (no extra text):
### Hypothesis N
**Statement:** <one-sentence hypothesis>
**Rationale:** <2–3 sentence explanation referencing the literature>
**Confidence:** <float between 0.0 and 1.0>
"""

_USER_HYPOTHESIS = """\
Research Topic: {topic}

Literature Summaries:
{summaries}

Generate {count} hypotheses.
"""


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_hypotheses(text: str, papers: List[Paper]) -> List[Hypothesis]:
    hypotheses: List[Hypothesis] = []

    # Split on "### Hypothesis N" markers
    blocks = re.split(r"###\s*Hypothesis\s+\d+", text, flags=re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        stmt_m = re.search(r"\*\*Statement:\*\*\s*(.+)", block)
        rat_m = re.search(r"\*\*Rationale:\*\*\s*([\s\S]+?)(?=\*\*Confidence|\Z)", block)
        conf_m = re.search(r"\*\*Confidence:\*\*\s*([0-9.]+)", block)

        statement = stmt_m.group(1).strip() if stmt_m else block[:120].strip()
        rationale = rat_m.group(1).strip() if rat_m else ""
        confidence = float(conf_m.group(1)) if conf_m else 0.5
        confidence = max(0.0, min(1.0, confidence))

        hypotheses.append(
            Hypothesis(
                statement=statement,
                rationale=rationale,
                supporting_papers=papers[:3],  # link top-3 papers
                confidence=confidence,
            )
        )

    return hypotheses


# ── Public API ────────────────────────────────────────────────────────────────

def suggest_hypotheses(
    papers: List[Paper],
    topic: str,
    count: Optional[int] = None,
) -> List[Hypothesis]:
    """
    Generate testable hypotheses from a set of papers.

    Parameters
    ----------
    papers:
        Papers to analyse (should already be summarised for best results).
    topic:
        Research topic string for context.
    count:
        Number of hypotheses to generate (defaults to config value).
    """
    count = count or config.hypothesis_count
    if not papers:
        logger.warning("No papers provided for hypothesis generation.")
        return []

    summaries = "\n\n".join(
        f"[{i+1}] {p.title} ({p.year or 'n.d.'})\n"
        f"{p.summary or p.abstract[:400]}"
        for i, p in enumerate(papers[:15])  # cap at 15 papers for context
    )

    system = _SYSTEM_HYPOTHESIS.format(count=count)
    user = _USER_HYPOTHESIS.format(
        topic=topic,
        summaries=summaries,
        count=count,
    )

    logger.info("Generating %d hypotheses for topic: %s", count, topic)
    raw = llm_client.generate(system, user, task_hint="hypothesis")
    hypotheses = _parse_hypotheses(raw, papers)

    # If parsing yielded fewer than requested (e.g., mock), pad gracefully
    if not hypotheses:
        hypotheses = [
            Hypothesis(
                statement=line.strip().lstrip("0123456789. "),
                rationale="Derived from synthesis of retrieved literature.",
                supporting_papers=papers[:3],
                confidence=0.5,
            )
            for line in raw.strip().splitlines()
            if line.strip()
        ]

    return hypotheses[:count]


# Avoid NameError — Optional is used in signature
from typing import Optional  # noqa: E402
