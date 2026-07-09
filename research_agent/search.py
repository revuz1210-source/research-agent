"""
Literature search module.

Providers (all free, no API key required by default):
  - ArxivProvider           — arXiv preprints (CS, Physics, Math, Bio, …)
  - CrossRefProvider        — 130 M+ published papers across all disciplines
  - EuropePMCProvider       — Biomedical / life-sciences full-text index
  - SemanticScholarProvider — Broad coverage; needs API key to avoid 429s

All implement ``BaseProvider`` and return ``List[Paper]``.
``search()`` fans out to all active providers and deduplicates results.
"""

from __future__ import annotations

import logging
import re
import ssl
import time
import urllib.parse
import urllib.request
import urllib.error
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import List, Optional

from research_agent.config import config
from research_agent.models import Author, Paper, SearchQuery

logger = logging.getLogger(__name__)

# ── SSL context ───────────────────────────────────────────────────────────────
# Windows Python installs often lack CA bundles.  We try the system certs first
# and fall back to an unverified context so network calls always work.
def _ssl_context() -> ssl.SSLContext:
    try:
        ctx = ssl.create_default_context()
        # quick probe — if this raises, fall back
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        return ctx
    except Exception:
        pass
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

_SSL_CTX = _ssl_context()


# ── HTTP helper with retry ────────────────────────────────────────────────────
_USER_AGENT = "ResearchAgent/1.0 (https://github.com/your-org/research-agent)"

def _fetch_url(url: str, headers: Optional[dict] = None, retries: int = 3) -> bytes:
    """
    Fetch a URL with:
      - SSL fallback (handles Windows cert issues)
      - User-Agent header (avoids 403 from some APIs)
      - Exponential back-off retry on 429 / 5xx
    """
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)

    req = urllib.request.Request(url, headers=hdrs)
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=config.request_timeout, context=_SSL_CTX) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                wait = 2 ** attempt          # 1 s, 2 s, 4 s
                logger.warning("Rate-limited (429) — retrying in %ds …", wait)
                time.sleep(wait)
                continue
            if exc.code >= 500:
                wait = 2 ** attempt
                logger.warning("Server error %d — retrying in %ds …", exc.code, wait)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as exc:
            last_exc = exc
            # SSL error on first attempt → switch to unverified and retry
            if "CERTIFICATE" in str(exc).upper() and attempt == 0:
                logger.warning("SSL cert error — retrying without verification.")
                _SSL_CTX.check_hostname = False
                _SSL_CTX.verify_mode = ssl.CERT_NONE
                continue
            raise

    raise last_exc

# ── Namespace map for arXiv Atom feed ─────────────────────────────────────────
_ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


# ── Base class ────────────────────────────────────────────────────────────────

class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def fetch(self, query: SearchQuery) -> List[Paper]:
        """Return a list of papers matching the query."""


# ── arXiv provider ────────────────────────────────────────────────────────────

class ArxivProvider(BaseProvider):
    name = "arxiv"
    _BASE = "https://export.arxiv.org/api/query"

    def fetch(self, query: SearchQuery) -> List[Paper]:
        # Use the raw query as a phrase search — most accurate and broadest.
        # Wrapping in quotes finds exact phrase first; fallback is keyword OR search.
        raw_q = query.raw.strip()
        search_terms = f'all:"{raw_q}"'
        max_r = query.max_results or config.arxiv_max_results

        params = urllib.parse.urlencode(
            {
                "search_query": search_terms,
                "max_results": max_r,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
        )
        url = f"{self._BASE}?{params}"
        logger.debug("arXiv request: %s", url)

        try:
            xml_data = _fetch_url(url)
        except Exception as exc:
            logger.warning("arXiv request failed: %s", exc)
            return []

        papers = self._parse(xml_data)

        # If phrase search returned nothing (topic not on arXiv), fall back to
        # keyword OR search so we always return something.
        if not papers and query.keywords:
            or_terms = " OR ".join(f"all:{kw}" for kw in query.keywords)
            params2 = urllib.parse.urlencode(
                {"search_query": or_terms, "max_results": max_r,
                 "sortBy": "relevance", "sortOrder": "descending"}
            )
            url2 = f"{self._BASE}?{params2}"
            logger.debug("arXiv fallback OR request: %s", url2)
            try:
                papers = self._parse(_fetch_url(url2))
            except Exception as exc:
                logger.warning("arXiv fallback failed: %s", exc)

        return papers

    def _parse(self, xml_data: bytes) -> List[Paper]:
        root = ET.fromstring(xml_data)
        papers: List[Paper] = []

        for entry in root.findall("atom:entry", _ARXIV_NS):
            paper_id_raw = (entry.findtext("atom:id", "", _ARXIV_NS) or "").strip()
            paper_id = paper_id_raw.split("/abs/")[-1] if "/abs/" in paper_id_raw else paper_id_raw

            title = (entry.findtext("atom:title", "", _ARXIV_NS) or "").strip()
            title = re.sub(r"\s+", " ", title)

            abstract = (entry.findtext("atom:summary", "", _ARXIV_NS) or "").strip()
            abstract = re.sub(r"\s+", " ", abstract)

            authors = [
                Author(name=(a.findtext("atom:name", "", _ARXIV_NS) or "").strip())
                for a in entry.findall("atom:author", _ARXIV_NS)
            ]

            published = entry.findtext("atom:published", "", _ARXIV_NS) or ""
            year = int(published[:4]) if len(published) >= 4 else None

            # arXiv DOI link
            doi: Optional[str] = None
            for link in entry.findall("atom:link", _ARXIV_NS):
                if link.get("title") == "doi":
                    doi = link.get("href")

            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    url=paper_id_raw,
                    doi=doi,
                    source="arxiv",
                )
            )

        return papers


# ── Semantic Scholar provider ─────────────────────────────────────────────────

class SemanticScholarProvider(BaseProvider):
    name = "semantic_scholar"
    _BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
    _FIELDS = "paperId,title,authors,abstract,year,venue,externalIds,citationCount,openAccessPdf"

    def fetch(self, query: SearchQuery) -> List[Paper]:
        # Always use the raw query — it's more precise than individual keywords
        q = query.raw
        params = urllib.parse.urlencode(
            {
                "query": q,
                "limit": query.max_results or config.semantic_scholar_max_results,
                "fields": self._FIELDS,
            }
        )
        url = f"{self._BASE}?{params}"
        logger.debug("Semantic Scholar request: %s", url)

        extra_headers = {}
        if config.semantic_scholar_api_key:
            extra_headers["x-api-key"] = config.semantic_scholar_api_key

        try:
            raw = _fetch_url(url, headers=extra_headers)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("Semantic Scholar request failed: %s", exc)
            return []

        return self._parse(data)

    def _parse(self, data: dict) -> List[Paper]:
        papers: List[Paper] = []
        for item in data.get("data", []):
            authors = [
                Author(name=a.get("name", ""))
                for a in item.get("authors", [])
            ]
            ext = item.get("externalIds") or {}
            doi = ext.get("DOI")
            pdf = item.get("openAccessPdf") or {}
            url = pdf.get("url")

            papers.append(
                Paper(
                    paper_id=item.get("paperId", ""),
                    title=item.get("title", ""),
                    authors=authors,
                    abstract=item.get("abstract") or "",
                    year=item.get("year"),
                    venue=item.get("venue"),
                    url=url,
                    doi=doi,
                    source="semantic_scholar",
                    citation_count=item.get("citationCount", 0),
                )
            )
        return papers


# ── CrossRef provider ─────────────────────────────────────────────────────────

class CrossRefProvider(BaseProvider):
    """
    Searches the CrossRef REST API — 130 M+ published works, no key required.
    https://api.crossref.org/swagger-ui/index.html
    """

    name = "crossref"
    _BASE = "https://api.crossref.org/works"

    def fetch(self, query: SearchQuery) -> List[Paper]:
        params = urllib.parse.urlencode(
            {
                "query": query.raw,
                "rows": min(query.max_results or 10, 20),
                "select": "DOI,title,author,abstract,published,container-title,is-referenced-by-count,URL",
            }
        )
        url = f"{self._BASE}?{params}"
        logger.debug("CrossRef request: %s", url)

        try:
            raw = _fetch_url(url)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("CrossRef request failed: %s", exc)
            return []

        return self._parse(data)

    def _parse(self, data: dict) -> List[Paper]:
        papers: List[Paper] = []
        for item in (data.get("message") or {}).get("items", []):
            # Title is a list
            title_list = item.get("title") or []
            title = title_list[0] if title_list else ""
            if not title:
                continue

            # Authors
            authors = []
            for a in item.get("author") or []:
                given = a.get("given", "")
                family = a.get("family", "")
                name = f"{given} {family}".strip() if given or family else a.get("name", "")
                if name:
                    authors.append(Author(name=name))

            # Year — try "published" date parts
            year: Optional[int] = None
            pub = item.get("published") or item.get("published-print") or {}
            parts = pub.get("date-parts", [[]])
            if parts and parts[0]:
                try:
                    year = int(parts[0][0])
                except (ValueError, TypeError):
                    pass

            doi = item.get("DOI", "")
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else None)
            venue_list = item.get("container-title") or []
            venue = venue_list[0] if venue_list else None
            abstract = item.get("abstract") or ""
            # Strip JATS XML tags CrossRef sometimes includes
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()

            papers.append(
                Paper(
                    paper_id=doi or title[:40],
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    venue=venue,
                    url=url,
                    doi=doi or None,
                    source="crossref",
                    citation_count=item.get("is-referenced-by-count", 0),
                )
            )
        return papers


# ── Europe PMC provider ───────────────────────────────────────────────────────

class EuropePMCProvider(BaseProvider):
    """
    Searches Europe PubMed Central — free, no key, biomedical & life sciences.
    https://europepmc.org/RestfulWebService
    """

    name = "europepmc"
    _BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def fetch(self, query: SearchQuery) -> List[Paper]:
        params = urllib.parse.urlencode(
            {
                "query": query.raw,
                "pageSize": min(query.max_results or 10, 25),
                "format": "json",
                "resultType": "core",
            }
        )
        url = f"{self._BASE}?{params}"
        logger.debug("Europe PMC request: %s", url)

        try:
            raw = _fetch_url(url)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("Europe PMC request failed: %s", exc)
            return []

        return self._parse(data)

    def _parse(self, data: dict) -> List[Paper]:
        papers: List[Paper] = []
        for item in (data.get("resultList") or {}).get("result", []):
            title = (item.get("title") or "").rstrip(".")
            if not title:
                continue

            # Author list
            authors = []
            for a in (item.get("authorList") or {}).get("author") or []:
                name = a.get("fullName") or a.get("lastName", "")
                if name:
                    authors.append(Author(name=name))

            year_str = item.get("pubYear") or ""
            year = int(year_str) if year_str.isdigit() else None

            doi = item.get("doi") or None
            pmid = item.get("pmid") or ""
            url = f"https://europepmc.org/article/{item.get('source','MED')}/{pmid}" if pmid else (
                f"https://doi.org/{doi}" if doi else None
            )
            abstract = (item.get("abstractText") or "").strip()
            venue = item.get("journalTitle") or item.get("bookOrReportDetails", {}).get("publisher")

            papers.append(
                Paper(
                    paper_id=item.get("id") or doi or title[:40],
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    venue=venue,
                    url=url,
                    doi=doi,
                    source="europepmc",
                    citation_count=int(item.get("citedByCount") or 0),
                )
            )
        return papers


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(papers: List[Paper]) -> List[Paper]:
    """Remove near-duplicate papers using normalised title matching."""
    seen: set[str] = set()
    unique: List[Paper] = []
    for p in papers:
        key = re.sub(r"\W+", "", p.title.lower())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ── Public API ────────────────────────────────────────────────────────────────

def search(
    query: SearchQuery,
    providers: Optional[List[str]] = None,
) -> List[Paper]:
    """
    Search all (or selected) providers and return a deduplicated, merged list.

    Parameters
    ----------
    query:
        Structured search query.
    providers:
        Subset of provider names to use (e.g. ``["arxiv"]``).
        Defaults to all available providers.
    """
    # Default free providers (no API key needed).
    # Semantic Scholar is opt-in only — it 429s unauthenticated IPs aggressively.
    _s2_has_key = bool(config.semantic_scholar_api_key)
    _s2_explicit = bool(providers and "semantic_scholar" in providers)

    _all: List[BaseProvider] = [
        ArxivProvider(),
        CrossRefProvider(),
        EuropePMCProvider(),
    ]
    if _s2_has_key or _s2_explicit:
        _all.append(SemanticScholarProvider())

    if providers:
        available = [p for p in _all if p.name in providers]
    else:
        available = _all

    all_papers: List[Paper] = []
    for provider in available:
        logger.info("Searching provider: %s", provider.name)
        results = provider.fetch(query)
        # Drop papers with no title or empty/missing abstract — they add noise
        results = [p for p in results if p.title.strip() and p.abstract.strip()]
        logger.info("  -> %d results (with abstracts)", len(results))
        all_papers.extend(results)

    deduplicated = _deduplicate(all_papers)

    # Apply date filter if requested
    if query.date_from or query.date_to:
        deduplicated = [
            p for p in deduplicated
            if p.year is not None
            and (query.date_from is None or p.year >= query.date_from)
            and (query.date_to is None or p.year <= query.date_to)
        ]

    # Sort by citation count descending (most-cited first)
    deduplicated.sort(key=lambda p: p.citation_count, reverse=True)
    return deduplicated
