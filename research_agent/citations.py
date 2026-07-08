"""
Citation management module.

Formats Paper objects into standard citation styles (APA, MLA, IEEE, Chicago)
and manages a reference library for a session.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from research_agent.config import config
from research_agent.models import Paper


# ── Formatters ────────────────────────────────────────────────────────────────

def _normalise_name(name: str):
    """Return (last, given_parts) from either 'Last, Given' or 'Given Last'."""
    name = name.strip()
    if "," in name:
        last, _, given = name.partition(",")
        return last.strip(), given.strip().split()
    parts = name.split()
    if len(parts) >= 2:
        return parts[-1], parts[:-1]
    return name, []


def _author_apa(names: List[str]) -> str:
    """Format author list in APA style (Last, F. I.)."""
    formatted = []
    for name in names:
        last, given_parts = _normalise_name(name)
        if given_parts:
            initials = " ".join(p[0] + "." for p in given_parts if p)
            formatted.append(f"{last}, {initials}")
        else:
            formatted.append(last)

    if len(formatted) == 0:
        return "Unknown Author"
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) <= 6:
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    # More than 6 authors → list first 6, then et al.
    return ", ".join(formatted[:6]) + ", … " + formatted[-1]


def _author_mla(names: List[str]) -> str:
    if not names:
        return "Unknown Author"
    first = names[0]
    parts = first.strip().split()
    first_formatted = (
        f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else first
    )
    if len(names) == 1:
        return first_formatted
    if len(names) == 2:
        return f"{first_formatted}, and {names[1]}"
    return f"{first_formatted}, et al."


def _author_ieee(names: List[str]) -> str:
    formatted = []
    for name in names:
        last, given_parts = _normalise_name(name)
        if given_parts:
            initials = ". ".join(p[0] for p in given_parts if p) + "."
            formatted.append(f"{initials} {last}")
        else:
            formatted.append(last)
    return ", ".join(formatted) if formatted else "Unknown"


def _author_chicago(names: List[str]) -> str:
    if not names:
        return "Unknown Author"
    last, given_parts = _normalise_name(names[0])
    first_formatted = (
        f"{last}, {' '.join(given_parts)}" if given_parts else last
    )
    rest = ", ".join(names[1:])
    return f"{first_formatted}{', and ' + rest if rest else ''}."


# ── Citation builders ─────────────────────────────────────────────────────────

def _cite_apa(p: Paper) -> str:
    names = [a.name for a in p.authors]
    author_str = _author_apa(names)
    year = p.year or "n.d."
    venue = f" *{p.venue}*." if p.venue else "."
    doi_str = f" https://doi.org/{p.doi}" if p.doi else (f" {p.url}" if p.url else "")
    return f"{author_str} ({year}). {p.title}{venue}{doi_str}"


def _cite_mla(p: Paper) -> str:
    names = [a.name for a in p.authors]
    author_str = _author_mla(names)
    year = p.year or "n.d."
    venue = f' *{p.venue}*' if p.venue else ""
    doi_str = f", {p.doi}" if p.doi else (f", {p.url}" if p.url else "")
    return f'{author_str} "{p.title}."{venue}{doi_str}, {year}.'


def _cite_ieee(p: Paper) -> str:
    names = [a.name for a in p.authors]
    author_str = _author_ieee(names)
    year = p.year or "n.d."
    venue = f', in *{p.venue}*' if p.venue else ""
    doi_str = f", doi: {p.doi}" if p.doi else ""
    return f'{author_str}, "{p.title}"{venue}, {year}{doi_str}.'


def _cite_chicago(p: Paper) -> str:
    names = [a.name for a in p.authors]
    author_str = _author_chicago(names)
    year = p.year or "n.d."
    venue = f" *{p.venue}*" if p.venue else ""
    doi_str = f" https://doi.org/{p.doi}" if p.doi else (f" {p.url}" if p.url else "")
    return f'{author_str} {year}. \u201c{p.title}.\u201d{venue}{doi_str}'


_FORMATTERS = {
    "apa": _cite_apa,
    "mla": _cite_mla,
    "ieee": _cite_ieee,
    "chicago": _cite_chicago,
}


# ── CitationManager class ─────────────────────────────────────────────────────

class CitationManager:
    """
    Manages a session-level reference library.

    Papers are keyed by their ``paper_id``.  The library can be serialised
    to/from a JSON file for persistence across sessions.
    """

    def __init__(self, style: Optional[str] = None) -> None:
        self.style: str = (style or config.citation_style).lower()
        self._library: Dict[str, Paper] = {}

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, paper: Paper) -> None:
        """Add a paper to the library (idempotent)."""
        self._library[paper.paper_id] = paper

    def add_many(self, papers: List[Paper]) -> None:
        for p in papers:
            self.add(p)

    def remove(self, paper_id: str) -> bool:
        return self._library.pop(paper_id, None) is not None

    def get(self, paper_id: str) -> Optional[Paper]:
        return self._library.get(paper_id)

    def all_papers(self) -> List[Paper]:
        return list(self._library.values())

    # ── Formatting ────────────────────────────────────────────────────────────

    def format(self, paper: Paper, style: Optional[str] = None) -> str:
        """Return a formatted citation string for a single paper."""
        style = (style or self.style).lower()
        formatter = _FORMATTERS.get(style, _cite_apa)
        return formatter(paper)

    def format_all(self, style: Optional[str] = None) -> List[str]:
        """Return sorted formatted citations for all library papers."""
        citations = [self.format(p, style) for p in self._library.values()]
        return sorted(citations)

    def bibliography(self, style: Optional[str] = None) -> str:
        """Return a newline-separated bibliography block."""
        return "\n\n".join(self.format_all(style))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the library to a JSON file."""
        data = {pid: p.to_dict() for pid, p in self._library.items()}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Restore a library from a JSON file (merges into current library)."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for pid, item in data.items():
            authors = [Author(name=n) for n in item.get("authors", [])]
            paper = Paper(
                paper_id=pid,
                title=item["title"],
                authors=authors,
                abstract=item.get("abstract", ""),
                year=item.get("year"),
                venue=item.get("venue"),
                url=item.get("url"),
                doi=item.get("doi"),
                source=item.get("source", "unknown"),
                citation_count=item.get("citation_count", 0),
                keywords=item.get("keywords", []),
                summary=item.get("summary"),
            )
            self._library[pid] = paper

    def __len__(self) -> int:
        return len(self._library)

    def __repr__(self) -> str:
        return f"CitationManager(style={self.style!r}, papers={len(self)})"


# Re-export Author for load() usage
from research_agent.models import Author  # noqa: E402 (avoid circular at top)
