"""
Paper summarisation module.

Given a Paper object, produces a concise academic summary using the LLM.
For batches of papers it also produces a comparative synthesis.
"""

from __future__ import annotations

import logging
from typing import List

from research_agent.config import config
from research_agent.models import Paper
from research_agent import llm_client

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_SUMMARISE = """\
You are an expert academic research assistant. Your task is to produce a concise, \
accurate summary of a research paper based on its title and abstract.

Guidelines:
- Length: {max_length} words or fewer
- Style: {style}
- Structure: one coherent paragraph
- Highlight: main objective, key methodology, principal findings, and significance
- Avoid padding or vague phrases like "the paper discusses"
"""

_USER_SUMMARISE = """\
Title: {title}

Abstract:
{abstract}

Provide a {style} summary.
"""

_SYSTEM_SYNTHESIS = """\
You are an expert academic research assistant. Synthesise the following set of \
paper summaries into a single coherent literature overview. \
Identify common themes, contradictions, and research gaps. \
Write in academic prose, 3–5 paragraphs.
"""

_USER_SYNTHESIS = """\
Research Topic: {topic}

Papers:
{papers_block}
"""


# ── Public API ────────────────────────────────────────────────────────────────

def summarise_paper(paper: Paper) -> str:
    """
    Generate a concise summary for a single paper.

    The result is stored in ``paper.summary`` and also returned.
    """
    if paper.summary:
        return paper.summary  # already summarised

    if not paper.abstract:
        paper.summary = f"No abstract available for: {paper.title}"
        return paper.summary

    system = _SYSTEM_SUMMARISE.format(
        max_length=config.summary_max_length,
        style=config.summary_style,
    )
    user = _USER_SUMMARISE.format(
        title=paper.title,
        abstract=paper.abstract,
        style=config.summary_style,
    )

    logger.debug("Summarising paper: %s", paper.title[:60])
    summary = llm_client.generate(system, user, task_hint="summarize")
    paper.summary = summary
    return summary


def summarise_papers(papers: List[Paper]) -> List[Paper]:
    """Summarise every paper in the list in-place and return the list."""
    for paper in papers:
        summarise_paper(paper)
    return papers


def synthesise(papers: List[Paper], topic: str) -> str:
    """
    Produce a synthesised literature overview from multiple paper summaries.

    All papers are summarised first if they lack a summary.
    """
    summarise_papers(papers)

    papers_block = "\n\n".join(
        f"[{i+1}] {p.title} ({p.year or 'n.d.'})\n{p.summary or p.abstract[:300]}"
        for i, p in enumerate(papers)
    )

    user = _USER_SYNTHESIS.format(topic=topic, papers_block=papers_block)
    logger.info("Generating literature synthesis for topic: %s", topic)
    return llm_client.generate(_SYSTEM_SYNTHESIS, user, task_hint="report_section")
