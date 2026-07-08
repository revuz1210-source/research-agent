"""
Report and paper-draft generation module.

Assembles a full ResearchReport and renders it as:
  - Markdown (human-readable)
  - JSON  (machine-readable)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from research_agent.config import config
from research_agent.models import Hypothesis, Paper, ResearchReport
from research_agent.citations import CitationManager
from research_agent import llm_client

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_INTRODUCTION = """\
You are an expert academic writer. Write a formal Introduction section for a \
research report on the given topic. The introduction should:
  - Motivate the research area
  - State the scope and objectives
  - Briefly outline what the report covers
  - Be 2–3 paragraphs, academic prose
"""

_SYSTEM_CONCLUSION = """\
You are an expert academic writer. Write a Conclusion section for a research report.
  - Summarise key findings from the reviewed literature
  - Highlight the most promising hypotheses
  - Identify limitations and directions for future work
  - Be 2–3 paragraphs, academic prose
"""

_USER_SECTION = """\
Topic: {topic}

Key Papers:
{papers_block}

Key Hypotheses:
{hyp_block}
"""


# ── Section generators ────────────────────────────────────────────────────────

def _papers_block(papers: List[Paper]) -> str:
    return "\n".join(
        f"  [{i+1}] {p.title} ({p.year or 'n.d.'}) — {p.authors[0].name if p.authors else 'Unknown'}"
        for i, p in enumerate(papers[:10])
    )


def _hyp_block(hypotheses: List[Hypothesis]) -> str:
    if not hypotheses:
        return "  (none generated)"
    return "\n".join(
        f"  H{i+1}: {h.statement}" for i, h in enumerate(hypotheses)
    )


def _generate_introduction(topic: str, papers: List[Paper], hyps: List[Hypothesis]) -> str:
    user = _USER_SECTION.format(
        topic=topic,
        papers_block=_papers_block(papers),
        hyp_block=_hyp_block(hyps),
    )
    return llm_client.generate(_SYSTEM_INTRODUCTION, user, task_hint="report_section")


def _generate_conclusion(topic: str, papers: List[Paper], hyps: List[Hypothesis]) -> str:
    user = _USER_SECTION.format(
        topic=topic,
        papers_block=_papers_block(papers),
        hyp_block=_hyp_block(hyps),
    )
    return llm_client.generate(_SYSTEM_CONCLUSION, user, task_hint="report_section")


def _generate_methods_note(papers: List[Paper]) -> str:
    """Summarise the methodological landscape of the retrieved papers."""
    methods_notes = [
        f"- {p.title}: {(p.summary or p.abstract)[:200]}"
        for p in papers[:8]
    ]
    return (
        "The following papers were retrieved and analysed using automated "
        "literature search (arXiv, Semantic Scholar) and NLP-based summarisation:\n\n"
        + "\n".join(methods_notes)
    )


# ── Main builder ──────────────────────────────────────────────────────────────

def build_report(
    query: str,
    topic: str,
    papers: List[Paper],
    hypotheses: List[Hypothesis],
    citation_style: Optional[str] = None,
) -> ResearchReport:
    """
    Assemble a full ResearchReport.

    Parameters
    ----------
    query:
        Original natural-language query.
    topic:
        Derived topic string.
    papers:
        Summarised papers.
    hypotheses:
        Generated hypotheses.
    citation_style:
        Override the global citation style.
    """
    mgr = CitationManager(style=citation_style or config.citation_style)
    mgr.add_many(papers)

    logger.info("Generating report sections …")
    introduction = _generate_introduction(topic, papers, hypotheses)
    methods_note = _generate_methods_note(papers)
    conclusion = _generate_conclusion(topic, papers, hypotheses)

    # Build literature-review section from paper summaries
    lit_review_parts = [
        f"### {p.title}\n\n{p.summary or p.abstract}"
        for p in papers
    ]
    literature_review = "\n\n".join(lit_review_parts) if lit_review_parts else "(No papers found.)"

    # Hypothesis section
    hyp_section_parts = []
    for i, h in enumerate(hypotheses):
        hyp_section_parts.append(
            f"**H{i+1}: {h.statement}**\n\n"
            f"*Rationale:* {h.rationale}\n\n"
            f"*Confidence:* {h.confidence:.2f}"
        )
    hypothesis_section = "\n\n---\n\n".join(hyp_section_parts) if hyp_section_parts else "(No hypotheses generated.)"

    sections = {
        "introduction": introduction,
        "literature_review": literature_review,
        "methods_note": methods_note,
        "hypotheses": hypothesis_section,
        "conclusion": conclusion,
    }

    return ResearchReport(
        title=f"Research Report: {topic.title()}",
        query=query,
        papers=papers,
        hypotheses=hypotheses,
        sections=sections,
        citations=mgr.format_all(),
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )


# ── Renderers ─────────────────────────────────────────────────────────────────

def render_markdown(report: ResearchReport) -> str:
    """Render a ResearchReport to a Markdown string."""
    lines = [
        f"# {report.title}",
        f"\n> Query: *{report.query}*",
        f"> Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"> Papers reviewed: {len(report.papers)}",
        "",
        "---",
        "",
        "## 1. Introduction",
        "",
        report.section("introduction"),
        "",
        "## 2. Literature Review",
        "",
        report.section("literature_review"),
        "",
        "## 3. Search & Summarisation Methodology",
        "",
        report.section("methods_note"),
        "",
        "## 4. Hypotheses",
        "",
        report.section("hypotheses"),
        "",
        "## 5. Conclusion",
        "",
        report.section("conclusion"),
        "",
        "---",
        "",
        "## References",
        "",
    ]
    for ref in report.citations:
        lines.append(f"- {ref}")

    return "\n".join(lines)


def render_json(report: ResearchReport) -> str:
    """Render a ResearchReport to a JSON string."""
    data = {
        "title": report.title,
        "query": report.query,
        "generated_at": report.generated_at.isoformat(),
        "papers": [p.to_dict() for p in report.papers],
        "hypotheses": [
            {
                "statement": h.statement,
                "rationale": h.rationale,
                "confidence": h.confidence,
            }
            for h in report.hypotheses
        ],
        "sections": report.sections,
        "citations": report.citations,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# ── File saver ────────────────────────────────────────────────────────────────

def save_report(
    report: ResearchReport,
    output_dir: Optional[str] = None,
    fmt: str = "markdown",
) -> str:
    """
    Save a report to disk.

    Returns the path to the saved file.
    """
    out_dir = Path(output_dir or config.report_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = report.generated_at.strftime("%Y%m%d_%H%M%S")
    slug = report.title.lower().replace(" ", "_").replace(":", "")[:40]
    ext = "md" if fmt == "markdown" else "json"
    path = out_dir / f"{slug}_{timestamp}.{ext}"

    content = render_markdown(report) if fmt == "markdown" else render_json(report)
    path.write_text(content, encoding="utf-8")
    logger.info("Report saved → %s", path)
    return str(path)
