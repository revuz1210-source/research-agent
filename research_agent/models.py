"""
Data models shared across all Research Agent modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Author:
    name: str
    affiliations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.name


@dataclass
class Paper:
    """Represents a single academic paper / preprint."""

    paper_id: str                              # provider-specific ID
    title: str
    authors: List[Author]
    abstract: str
    year: Optional[int] = None
    venue: Optional[str] = None               # journal / conference
    url: Optional[str] = None
    doi: Optional[str] = None
    source: str = "unknown"                   # arxiv | semantic_scholar | …
    citation_count: int = 0
    keywords: List[str] = field(default_factory=list)

    # Populated after summarisation
    summary: Optional[str] = None

    # ── Helpers ───────────────────────────────────────────────────────────────
    def short_ref(self) -> str:
        """One-line bibliographic reference."""
        first_author = self.authors[0].name if self.authors else "Unknown"
        year = self.year or "n.d."
        return f"{first_author} ({year}). {self.title}."

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": [a.name for a in self.authors],
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "doi": self.doi,
            "source": self.source,
            "citation_count": self.citation_count,
            "keywords": self.keywords,
            "summary": self.summary,
        }


@dataclass
class SearchQuery:
    """Structured representation of a research query."""

    raw: str
    keywords: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    date_from: Optional[int] = None           # year
    date_to: Optional[int] = None             # year
    max_results: int = 10


@dataclass
class Hypothesis:
    """A generated research hypothesis."""

    statement: str
    rationale: str
    supporting_papers: List[Paper] = field(default_factory=list)
    confidence: float = 0.0                   # 0.0–1.0

    def __str__(self) -> str:
        return f"Hypothesis: {self.statement}\nRationale: {self.rationale}"


@dataclass
class ResearchReport:
    """A fully compiled research report."""

    title: str
    query: str
    papers: List[Paper]
    hypotheses: List[Hypothesis]
    sections: dict                             # section_name → content
    citations: List[str]                      # formatted citation strings
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def section(self, name: str) -> str:
        return self.sections.get(name, "")
