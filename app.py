"""
FastAPI REST API wrapper for the Research Agent.

Run locally:
    pip install fastapi uvicorn
    uvicorn app:app --reload

Deploy to Render / Railway / Fly.io:
    - Connect your GitHub repo
    - Set start command: uvicorn app:app --host 0.0.0.0 --port $PORT
    - Add OPENAI_API_KEY as an environment variable

Endpoints:
    POST /research        — full pipeline, returns JSON report
    POST /search          — search only, returns list of papers
    POST /summarise       — summarise a list of papers
    POST /hypotheses      — generate hypotheses from papers
    GET  /health          — liveness check
    GET  /docs            — interactive Swagger UI (auto-generated)
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from research_agent.agent import ResearchAgent
from research_agent import reporter as rep_module

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Research Agent API",
    description="AI-powered academic research: search, summarise, hypothesise, report.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    question: str = Field(..., example="transformer models for protein folding")
    max_results: int = Field(10, ge=1, le=50)
    citation_style: str = Field("apa", pattern="^(apa|mla|ieee|chicago)$")
    providers: Optional[List[str]] = Field(None, example=["arxiv", "semantic_scholar"])
    generate_hypotheses: bool = True
    format: str = Field("json", pattern="^(markdown|json)$")


class SearchRequest(BaseModel):
    question: str
    max_results: int = Field(10, ge=1, le=50)
    providers: Optional[List[str]] = None


class PaperIn(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int] = None
    source: str = "unknown"
    summary: Optional[str] = None


class HypothesisRequest(BaseModel):
    papers: List[PaperIn]
    topic: str
    count: int = Field(3, ge=1, le=10)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _paper_in_to_model(p: PaperIn):
    from research_agent.models import Author, Paper
    return Paper(
        paper_id=p.paper_id,
        title=p.title,
        authors=[Author(name=n) for n in p.authors],
        abstract=p.abstract,
        year=p.year,
        source=p.source,
        summary=p.summary,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health():
    """Liveness check."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/research", tags=["Pipeline"])
def research(req: ResearchRequest):
    """Run the full research pipeline and return a structured report."""
    try:
        agent = ResearchAgent(
            citation_style=req.citation_style,
            providers=req.providers,
        )
        report = agent.run(
            raw_query=req.question,
            max_results=req.max_results,
            generate_hypotheses=req.generate_hypotheses,
            save_report=False,
        )
        if req.format == "markdown":
            return {"format": "markdown", "content": rep_module.render_markdown(report)}

        return {
            "title": report.title,
            "query": report.query,
            "generated_at": report.generated_at.isoformat(),
            "paper_count": len(report.papers),
            "papers": [p.to_dict() for p in report.papers],
            "hypotheses": [
                {"statement": h.statement, "rationale": h.rationale, "confidence": h.confidence}
                for h in report.hypotheses
            ],
            "sections": report.sections,
            "citations": report.citations,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/search", tags=["Steps"])
def search(req: SearchRequest):
    """Search literature and return a list of papers (no summarisation)."""
    try:
        agent = ResearchAgent(providers=req.providers)
        papers = agent.search(req.question, max_results=req.max_results)
        return {"count": len(papers), "papers": [p.to_dict() for p in papers]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/hypotheses", tags=["Steps"])
def hypotheses(req: HypothesisRequest):
    """Generate hypotheses from a provided list of papers."""
    try:
        papers = [_paper_in_to_model(p) for p in req.papers]
        agent = ResearchAgent()
        hyps = agent.suggest_hypotheses(papers, topic=req.topic, count=req.count)
        return {
            "count": len(hyps),
            "hypotheses": [
                {"statement": h.statement, "rationale": h.rationale, "confidence": h.confidence}
                for h in hyps
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
