"""
Research Agent Orchestrator.

``ResearchAgent`` is the single entry-point that wires together all modules:
  1. Parse natural-language query → SearchQuery
  2. Search literature (arXiv + Semantic Scholar)
  3. Summarise each paper
  4. Add papers to the citation library
  5. Suggest hypotheses
  6. Compile and save the full research report

Usage example:
    from research_agent.agent import ResearchAgent

    agent = ResearchAgent()
    report = agent.run("What are the latest advances in transformer-based protein folding?")
    print(report.section("conclusion"))
"""

from __future__ import annotations

import logging
from typing import List, Optional

from research_agent.config import Config, config as default_config
from research_agent.models import Hypothesis, Paper, ResearchReport, SearchQuery
from research_agent.query_parser import parse_query
from research_agent import search as search_module
from research_agent import summariser
from research_agent.citations import CitationManager
from research_agent import hypothesis as hypothesis_module
from research_agent import reporter

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Central orchestrator for the Research Agent pipeline.

    Parameters
    ----------
    config:
        Configuration object.  Defaults to the shared singleton.
    citation_style:
        Override the default citation style (apa | mla | ieee | chicago).
    providers:
        Override the default search providers (e.g. ``["arxiv"]``).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        citation_style: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.config = config or default_config
        self.citation_manager = CitationManager(
            style=citation_style or self.config.citation_style
        )
        self.providers = providers  # None → all providers

        # Session state — populated after run()
        self.last_query: Optional[SearchQuery] = None
        self.last_papers: List[Paper] = []
        self.last_hypotheses: List[Hypothesis] = []
        self.last_report: Optional[ResearchReport] = None

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        raw_query: str,
        max_results: int = 10,
        generate_hypotheses: bool = True,
        save_report: bool = True,
        report_format: str = "markdown",
        output_dir: Optional[str] = None,
    ) -> ResearchReport:
        """
        Run the full research pipeline end-to-end.

        Parameters
        ----------
        raw_query:
            Free-form research question.
        max_results:
            Maximum number of papers to retrieve per provider.
        generate_hypotheses:
            Whether to run the hypothesis-generation step.
        save_report:
            Whether to persist the report to disk.
        report_format:
            Output format: ``"markdown"`` or ``"json"``.
        output_dir:
            Override the configured output directory.

        Returns
        -------
        ResearchReport
        """
        logger.info("═" * 60)
        logger.info("Research Agent — starting pipeline")
        logger.info("Query: %s", raw_query)

        # Step 1 — Parse query
        logger.info("Step 1/5 — Parsing query …")
        query = parse_query(raw_query, max_results=max_results, use_llm=True)
        self.last_query = query
        logger.info("Keywords: %s", ", ".join(query.keywords))

        # Step 2 — Search literature
        logger.info("Step 2/5 — Searching literature …")
        papers = search_module.search(query, providers=self.providers)
        self.last_papers = papers
        logger.info("Retrieved %d unique papers.", len(papers))

        # Step 3 — Summarise papers
        logger.info("Step 3/5 — Summarising papers …")
        summariser.summarise_papers(papers)
        self.citation_manager.add_many(papers)

        # Step 4 — Generate hypotheses
        hypotheses: List[Hypothesis] = []
        if generate_hypotheses:
            logger.info("Step 4/5 — Generating hypotheses …")
            hypotheses = hypothesis_module.suggest_hypotheses(
                papers, topic=query.topic or raw_query
            )
            self.last_hypotheses = hypotheses
            logger.info("Generated %d hypotheses.", len(hypotheses))
        else:
            logger.info("Step 4/5 — Hypothesis generation skipped.")

        # Step 5 — Build and optionally save report
        logger.info("Step 5/5 — Building report …")
        report = reporter.build_report(
            query=raw_query,
            topic=query.topic or raw_query,
            papers=papers,
            hypotheses=hypotheses,
            citation_style=self.citation_manager.style,
        )
        self.last_report = report

        if save_report:
            path = reporter.save_report(
                report,
                output_dir=output_dir or self.config.report_output_dir,
                fmt=report_format,
            )
            logger.info("Report saved → %s", path)

        logger.info("Pipeline complete. Papers: %d | Hypotheses: %d", len(papers), len(hypotheses))
        logger.info("═" * 60)
        return report

    # ── Standalone helpers ────────────────────────────────────────────────────

    def search(self, raw_query: str, max_results: int = 10) -> List[Paper]:
        """Run only the search step and return papers."""
        query = parse_query(raw_query, max_results=max_results)
        return search_module.search(query, providers=self.providers)

    def summarise(self, papers: List[Paper]) -> List[Paper]:
        """Summarise a list of papers and return them."""
        return summariser.summarise_papers(papers)

    def synthesise(self, papers: List[Paper], topic: str) -> str:
        """Return a synthesised literature overview string."""
        return summariser.synthesise(papers, topic)

    def suggest_hypotheses(
        self,
        papers: List[Paper],
        topic: str,
        count: Optional[int] = None,
    ) -> List[Hypothesis]:
        """Run only the hypothesis step."""
        return hypothesis_module.suggest_hypotheses(papers, topic, count=count)

    def bibliography(self, style: Optional[str] = None) -> str:
        """Return a formatted bibliography of all session papers."""
        return self.citation_manager.bibliography(style)

    def save_citations(self, path: str) -> None:
        """Persist the citation library to a JSON file."""
        self.citation_manager.save(path)
        logger.info("Citations saved → %s", path)

    def load_citations(self, path: str) -> None:
        """Load a previously saved citation library."""
        self.citation_manager.load(path)
        logger.info("Citations loaded ← %s", path)
