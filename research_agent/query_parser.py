"""
Query parser — converts a natural-language research question into a structured
SearchQuery using lightweight heuristics and (optionally) an LLM keyword extractor.
"""

from __future__ import annotations

import re
from typing import List, Optional

from research_agent.config import config
from research_agent.models import SearchQuery
from research_agent import llm_client


# ── Stop-words (minimal set for keyword extraction) ───────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "that", "this", "these", "those", "i", "we", "you", "he", "she", "it",
    "they", "me", "him", "her", "us", "them", "my", "our", "your", "his",
    "about", "between", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "research", "study", "paper", "papers", "article", "articles",
    "work", "works", "using", "use", "based", "new", "novel", "recent",
}


def _heuristic_keywords(text: str) -> List[str]:
    """Extract significant words by stripping stop-words and short tokens."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]*", text.lower())
    seen: dict[str, int] = {}
    for tok in tokens:
        if tok not in _STOP_WORDS and len(tok) > 2:
            seen[tok] = seen.get(tok, 0) + 1
    # Return unique keywords sorted by frequency, top 8
    return [kw for kw, _ in sorted(seen.items(), key=lambda x: -x[1])][:8]


def _llm_keywords(text: str) -> List[str]:
    """
    Use the LLM to extract domain-specific keywords.
    Returns empty list if the LLM is unavailable or returns non-keyword prose,
    so the caller always falls back to heuristic extraction.
    """
    system = (
        "You are a research librarian. Extract the most important academic "
        "keywords from the user's query. Return ONLY a semicolon-separated list "
        "of keywords with no extra text. Example: protein folding; transformer; AlphaFold"
    )
    try:
        raw = llm_client.generate_strict(system, text)
    except Exception:
        return []

    candidates = [kw.strip() for kw in raw.split(";") if kw.strip()]

    # Sanity check: real keywords are short (≤5 words each) and there are
    # multiple of them. If the LLM returned a prose sentence, discard it.
    if not candidates:
        return []
    if len(candidates) == 1 and len(candidates[0].split()) > 6:
        return []   # looks like a sentence, not keywords
    # Filter out any individual "keyword" that is suspiciously long
    return [kw for kw in candidates if len(kw.split()) <= 5][:8]


def _extract_year_range(text: str):
    """Parse optional 'from YYYY to YYYY' or 'since YYYY' patterns."""
    date_from: Optional[int] = None
    date_to: Optional[int] = None

    since = re.search(r"\bsince\s+(\d{4})\b", text, re.I)
    if since:
        date_from = int(since.group(1))

    between = re.search(r"\bfrom\s+(\d{4})\s+to\s+(\d{4})\b", text, re.I)
    if between:
        date_from, date_to = int(between.group(1)), int(between.group(2))

    return date_from, date_to


# ── Public API ────────────────────────────────────────────────────────────────

def parse_query(
    raw: str,
    max_results: int = 10,
    use_llm: bool = True,
) -> SearchQuery:
    """
    Parse a natural-language research question into a ``SearchQuery``.

    Parameters
    ----------
    raw:
        Free-form research question or topic string.
    max_results:
        Maximum number of results to fetch from each provider.
    use_llm:
        Whether to call the LLM for keyword refinement.
    """
    date_from, date_to = _extract_year_range(raw)

    # Keyword extraction: always use heuristic first (works offline, query-specific),
    # then optionally enrich with LLM when an API key is available.
    keywords = _heuristic_keywords(raw)
    if use_llm and config.openai_api_key:
        llm_kws = _llm_keywords(raw)
        if llm_kws:
            keywords = llm_kws

    # Derive topic as the first 3 keywords joined
    topic = " ".join(keywords[:3]) if keywords else raw[:60]

    return SearchQuery(
        raw=raw,
        keywords=keywords,
        topic=topic,
        date_from=date_from,
        date_to=date_to,
        max_results=max_results,
    )
