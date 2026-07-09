"""
Streamlit web UI for the Research Agent.

Run locally:
    pip install streamlit
    streamlit run streamlit_app.py

Deploy free:
    Push to GitHub → https://share.streamlit.io → connect repo
    Add OPENAI_API_KEY in the Secrets section.
"""

import json
import os

import streamlit as st

# ── Load secrets (Streamlit Cloud) and .env (local) ──────────────────────────
def _load_secrets():
    try:
        for key in ("OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"):
            val = st.secrets.get(key, "")
            if val and not os.environ.get(key):
                os.environ[key] = val
    except Exception:
        pass

_load_secrets()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state — persists across every Streamlit rerun ────────────────────
# This is the KEY fix: results are stored here so switching tabs / changing
# sidebar settings never wipes the displayed papers.
if "report"       not in st.session_state: st.session_state.report       = None
if "last_query"   not in st.session_state: st.session_state.last_query   = ""
if "running"      not in st.session_state: st.session_state.running      = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    _env_key = os.environ.get("OPENAI_API_KEY", "")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=_env_key,
        type="password",
        placeholder="sk-… (leave blank for mock mode)",
        help="Leave blank to use free search (no AI summaries). Set once in your .env file.",
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        max_results = st.number_input("Max papers", min_value=1, max_value=50, value=8)
    with col2:
        citation_style = st.selectbox("Citation style", ["apa", "ieee", "mla", "chicago"])

    providers = st.multiselect(
        "Search providers",
        options=["arxiv", "crossref", "europepmc", "semantic_scholar"],
        default=["arxiv", "crossref", "europepmc"],
        help="arXiv, CrossRef and Europe PMC are free — no key needed.",
    )

    generate_hyp = st.toggle("Generate hypotheses", value=True)
    report_fmt   = st.radio("Export format", ["markdown", "json"], horizontal=True)

    st.divider()

    # Show which query produced the current results
    if st.session_state.last_query:
        st.caption(f"Showing results for: **{st.session_state.last_query}**")

    if st.session_state.report:
        if st.button("🗑️ Clear results", use_container_width=True):
            st.session_state.report     = None
            st.session_state.last_query = ""
            st.rerun()

    st.caption(
        "📖 **Mock mode** — runs without an API key. "
        "Real papers are always fetched; AI text is placeholder only."
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔬 Research Agent")
st.caption("AI-powered literature search · summarisation · hypothesis generation · report writing")

query = st.text_input(
    "Enter your research question or topic",
    placeholder="e.g.  CRISPR gene editing safety",
    key="query_input",
)

run_btn = st.button("🚀 Run Research", type="primary", use_container_width=True)

# ── Update env key on every rerun so it's always current ─────────────────────
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
elif not openai_key:
    os.environ.pop("OPENAI_API_KEY", None)

# ── Run pipeline when button clicked ─────────────────────────────────────────
if run_btn:
    if not query.strip():
        st.warning("Please enter a research question first.")
        st.stop()

    # Clear previous results immediately so stale data is never shown
    st.session_state.report     = None
    st.session_state.last_query = query.strip()

    from research_agent.agent import ResearchAgent
    from research_agent import reporter as rep_module

    agent = ResearchAgent(
        citation_style=citation_style,
        providers=providers if providers else None,
    )

    progress = st.progress(0, text="Starting pipeline…")
    status   = st.empty()

    def tick(pct: int, msg: str):
        progress.progress(pct, text=msg)
        status.caption(msg)

    tick(10,  "🔍 Parsing query & extracting keywords…")
    tick(20,  "🌐 Searching arXiv, CrossRef, Europe PMC…")

    with st.spinner(f'Searching for: "{query.strip()}" — please wait…'):
        try:
            report = agent.run(
                raw_query=query.strip(),
                max_results=int(max_results),
                generate_hypotheses=generate_hyp,
                save_report=False,
            )
            # Store in session state — survives tab switches and sidebar changes
            st.session_state.report = report
            tick(100, "✅ Done!")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            progress.empty()
            status.empty()
            st.stop()

    progress.empty()
    status.empty()
    # Force a clean rerun so the results render fresh with no stale widgets
    st.rerun()

# ── Display results from session state ───────────────────────────────────────
report = st.session_state.report

if report is not None:
    from research_agent import reporter as rep_module

    st.success(
        f'Results for: **"{st.session_state.last_query}"** — '
        f'**{len(report.papers)} papers** · **{len(report.hypotheses)} hypotheses**'
    )

    tab_intro, tab_lit, tab_hyp, tab_concl, tab_refs, tab_export = st.tabs([
        "📖 Introduction",
        "📚 Literature Review",
        "💡 Hypotheses",
        "🏁 Conclusion",
        "📋 References",
        "⬇️ Export",
    ])

    with tab_intro:
        st.markdown(report.section("introduction") or "_No introduction generated._")

    with tab_lit:
        st.markdown(report.section("literature_review") or "_No literature review generated._")
        st.divider()
        st.subheader(f"Papers Retrieved ({len(report.papers)})")
        for i, p in enumerate(report.papers, 1):
            with st.expander(f"{i}. {p.title} ({p.year or 'n.d.'})"):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.caption(f"**Authors:** {', '.join(a.name for a in p.authors[:5])}")
                    st.caption(
                        f"**Venue:** {p.venue or '—'}  |  "
                        f"**Citations:** {p.citation_count}  |  "
                        f"**Source:** {p.source}"
                    )
                    if p.url:
                        st.markdown(f"[🔗 View paper]({p.url})")
                with cols[1]:
                    st.metric("Citations", p.citation_count)
                st.markdown("**Abstract**")
                st.write(p.abstract or "_No abstract available._")
                if p.summary:
                    st.markdown("**AI Summary**")
                    st.info(p.summary)

    with tab_hyp:
        if report.hypotheses:
            for i, h in enumerate(report.hypotheses, 1):
                with st.container(border=True):
                    st.markdown(f"**H{i}: {h.statement}**")
                    st.caption(f"Rationale: {h.rationale}")
                    st.progress(h.confidence, text=f"Confidence: {h.confidence:.0%}")
        else:
            st.info("Hypothesis generation was disabled or returned no results.")

    with tab_concl:
        st.markdown(report.section("conclusion") or "_No conclusion generated._")

    with tab_refs:
        for ref in report.citations:
            st.markdown(f"- {ref}")

    with tab_export:
        st.subheader("Download Report")
        if report_fmt == "markdown":
            content = rep_module.render_markdown(report)
            st.download_button(
                "⬇️ Download Markdown",
                data=content.encode(),
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
            with st.expander("Preview"):
                st.code(content[:3000] + ("…" if len(content) > 3000 else ""), language="markdown")
        else:
            content = rep_module.render_json(report)
            st.download_button(
                "⬇️ Download JSON",
                data=content.encode(),
                file_name="research_report.json",
                mime="application/json",
                use_container_width=True,
            )
            with st.expander("Preview"):
                st.json(json.loads(content))

else:
    # Landing page — no results yet
    st.info(
        "👆 Type a research topic above and click **Run Research**.\n\n"
        "**Try these:**\n"
        "- *CRISPR gene editing safety*\n"
        "- *black hole neutron star mergers*\n"
        "- *ocean microplastics pollution*\n"
        "- *quantum error correction surface codes*\n"
        "- *transformer protein structure prediction*"
    )
