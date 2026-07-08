"""
Tests for the Research Agent — no network calls, no LLM API key required.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_paper(n: int = 1):
    from research_agent.models import Author, Paper
    return Paper(
        paper_id=f"test_{n}",
        title=f"Test Paper {n}: Advances in AI Research",
        authors=[Author(name=f"Smith, J."), Author(name=f"Doe, A.")],
        abstract=(
            "This paper presents a comprehensive study on AI research methods. "
            "We propose novel techniques and validate them empirically. "
            "Results show significant improvements over baselines."
        ),
        year=2023 + n,
        venue="Journal of Artificial Intelligence",
        url=f"https://arxiv.org/abs/2401.0000{n}",
        doi=f"10.1234/test.{n}",
        source="arxiv",
        citation_count=100 * n,
    )


# ── query_parser ──────────────────────────────────────────────────────────────

class TestQueryParser:
    def test_heuristic_keywords(self):
        from research_agent.query_parser import _heuristic_keywords
        kws = _heuristic_keywords("deep learning for natural language processing")
        assert "deep" in kws or "learning" in kws or "natural" in kws

    def test_parse_query_returns_search_query(self):
        from research_agent.query_parser import parse_query
        q = parse_query("protein folding with transformers", use_llm=False)
        assert q.raw == "protein folding with transformers"
        assert len(q.keywords) > 0

    def test_year_range_since(self):
        from research_agent.query_parser import parse_query
        q = parse_query("neural networks since 2020", use_llm=False)
        assert q.date_from == 2020

    def test_year_range_between(self):
        from research_agent.query_parser import parse_query
        q = parse_query("climate models from 2010 to 2022", use_llm=False)
        assert q.date_from == 2010
        assert q.date_to == 2022

    def test_max_results_propagated(self):
        from research_agent.query_parser import parse_query
        q = parse_query("test query", max_results=25, use_llm=False)
        assert q.max_results == 25


# ── models ────────────────────────────────────────────────────────────────────

class TestModels:
    def test_paper_short_ref(self):
        p = _make_paper(1)
        ref = p.short_ref()
        assert "Smith" in ref
        assert "2024" in ref

    def test_paper_to_dict_keys(self):
        p = _make_paper(1)
        d = p.to_dict()
        for key in ["paper_id", "title", "authors", "abstract", "year", "url"]:
            assert key in d

    def test_hypothesis_str(self):
        from research_agent.models import Hypothesis
        h = Hypothesis(statement="X causes Y", rationale="Because Z.")
        assert "X causes Y" in str(h)


# ── citations ─────────────────────────────────────────────────────────────────

class TestCitationManager:
    def test_apa_format(self):
        from research_agent.citations import CitationManager
        mgr = CitationManager(style="apa")
        p = _make_paper(1)
        cite = mgr.format(p)
        assert "Smith" in cite
        assert "2024" in cite
        assert p.title in cite

    def test_ieee_format(self):
        from research_agent.citations import CitationManager
        mgr = CitationManager(style="ieee")
        p = _make_paper(1)
        cite = mgr.format(p)
        assert "Smith" in cite

    def test_mla_format(self):
        from research_agent.citations import CitationManager
        mgr = CitationManager(style="mla")
        p = _make_paper(1)
        cite = mgr.format(p)
        assert p.title in cite

    def test_chicago_format(self):
        from research_agent.citations import CitationManager
        mgr = CitationManager(style="chicago")
        p = _make_paper(1)
        cite = mgr.format(p)
        assert "2024" in cite

    def test_add_and_len(self):
        from research_agent.citations import CitationManager
        mgr = CitationManager()
        mgr.add(_make_paper(1))
        mgr.add(_make_paper(2))
        assert len(mgr) == 2

    def test_remove(self):
        from research_agent.citations import CitationManager
        mgr = CitationManager()
        p = _make_paper(1)
        mgr.add(p)
        removed = mgr.remove("test_1")
        assert removed is True
        assert len(mgr) == 0

    def test_save_and_load(self):
        from research_agent.citations import CitationManager
        mgr = CitationManager()
        mgr.add(_make_paper(1))
        mgr.add(_make_paper(2))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr.save(path)
            mgr2 = CitationManager()
            mgr2.load(path)
            assert len(mgr2) == 2
        finally:
            os.unlink(path)

    def test_format_all_sorted(self):
        from research_agent.citations import CitationManager
        mgr = CitationManager(style="apa")
        for i in range(3, 0, -1):
            mgr.add(_make_paper(i))
        cites = mgr.format_all()
        assert cites == sorted(cites)


# ── summariser ────────────────────────────────────────────────────────────────

class TestSummariser:
    def test_summarise_paper_no_abstract(self):
        from research_agent.models import Author, Paper
        from research_agent.summariser import summarise_paper
        p = Paper(
            paper_id="empty", title="Empty Paper",
            authors=[Author(name="Nobody")], abstract="",
        )
        summary = summarise_paper(p)
        assert "No abstract" in summary

    def test_summarise_paper_sets_summary(self):
        from research_agent.summariser import summarise_paper
        p = _make_paper(1)
        with patch("research_agent.llm_client.generate", return_value="Mock summary."):
            summary = summarise_paper(p)
        assert summary == "Mock summary."
        assert p.summary == "Mock summary."

    def test_summarise_paper_caches(self):
        from research_agent.summariser import summarise_paper
        p = _make_paper(1)
        p.summary = "Already done."
        with patch("research_agent.llm_client.generate") as mock_llm:
            result = summarise_paper(p)
        mock_llm.assert_not_called()
        assert result == "Already done."


# ── hypothesis ────────────────────────────────────────────────────────────────

class TestHypothesis:
    def test_returns_list(self):
        from research_agent.hypothesis import suggest_hypotheses
        papers = [_make_paper(i) for i in range(3)]
        for p in papers:
            p.summary = "Mock summary."
        with patch("research_agent.llm_client.generate", return_value=(
            "### Hypothesis 1\n"
            "**Statement:** AI will surpass human performance.\n"
            "**Rationale:** Based on recent scaling laws.\n"
            "**Confidence:** 0.7\n"
        )):
            hyps = suggest_hypotheses(papers, topic="AI scaling", count=1)
        assert len(hyps) >= 1
        assert "AI will surpass" in hyps[0].statement

    def test_empty_papers(self):
        from research_agent.hypothesis import suggest_hypotheses
        hyps = suggest_hypotheses([], topic="test")
        assert hyps == []


# ── search deduplication ──────────────────────────────────────────────────────

class TestSearchDeduplicate:
    def test_deduplication(self):
        from research_agent.search import _deduplicate
        p1 = _make_paper(1)
        p2 = _make_paper(1)  # same title
        p2.paper_id = "duplicate"
        result = _deduplicate([p1, p2])
        assert len(result) == 1

    def test_no_duplicates_unchanged(self):
        from research_agent.search import _deduplicate
        papers = [_make_paper(i) for i in range(4)]
        result = _deduplicate(papers)
        assert len(result) == 4


# ── reporter ─────────────────────────────────────────────────────────────────

class TestReporter:
    def _build_report(self):
        from research_agent.reporter import build_report
        papers = [_make_paper(i) for i in range(3)]
        for p in papers:
            p.summary = "Summary text."
        with patch("research_agent.llm_client.generate", return_value="Generated section."):
            return build_report(
                query="AI research",
                topic="artificial intelligence",
                papers=papers,
                hypotheses=[],
            )

    def test_report_has_sections(self):
        report = self._build_report()
        for key in ["introduction", "literature_review", "conclusion"]:
            assert key in report.sections

    def test_render_markdown(self):
        from research_agent.reporter import render_markdown
        report = self._build_report()
        md = render_markdown(report)
        assert "# Research Report" in md
        assert "## References" in md

    def test_render_json(self):
        from research_agent.reporter import render_json
        report = self._build_report()
        data = json.loads(render_json(report))
        assert "title" in data
        assert "papers" in data
        assert "citations" in data

    def test_save_report_creates_file(self):
        from research_agent.reporter import save_report
        report = self._build_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_report(report, output_dir=tmpdir, fmt="markdown")
            assert os.path.isfile(path)
            content = open(path).read()
            assert "# Research Report" in content


# ── agent integration ─────────────────────────────────────────────────────────

class TestAgentIntegration:
    def test_agent_run_offline(self):
        """Full pipeline with no network calls — uses mocked search and LLM."""
        from research_agent.agent import ResearchAgent

        mock_papers = [_make_paper(i) for i in range(3)]

        with (
            patch("research_agent.search.ArxivProvider.fetch", return_value=mock_papers),
            patch("research_agent.search.CrossRefProvider.fetch", return_value=[]),
            patch("research_agent.search.EuropePMCProvider.fetch", return_value=[]),
            patch("research_agent.llm_client.generate", return_value="Mocked LLM output."),
        ):
            agent = ResearchAgent()
            report = agent.run(
                "deep learning protein folding",
                max_results=5,
                save_report=False,
            )

        assert report is not None
        assert len(report.papers) == 3
        assert report.section("introduction") != ""

    def test_agent_search_only(self):
        from research_agent.agent import ResearchAgent

        mock_papers = [_make_paper(i) for i in range(2)]
        with (
            patch("research_agent.search.ArxivProvider.fetch", return_value=mock_papers),
            patch("research_agent.search.CrossRefProvider.fetch", return_value=[]),
            patch("research_agent.search.EuropePMCProvider.fetch", return_value=[]),
            patch("research_agent.llm_client.generate", return_value="keywords"),
        ):
            agent = ResearchAgent()
            papers = agent.search("quantum error correction", max_results=5)
        assert len(papers) == 2
