"""
Streamlit web UI for the Research Agent.

Run locally:
    pip install streamlit
    streamlit run streamlit_app.py

Deploy free:
    Push to GitHub → https://share.streamlit.io → connect repo
    Add OPENAI_API_KEY in the Secrets section.
"""

import io
import json

import os
import streamlit as st

# ── Inject secrets from Streamlit Cloud into env (no-op locally) ──────────────
def _load_streamlit_secrets():
    try:
        import streamlit as _st
        for key in ("OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"):
            val = _st.secrets.get(key, "")
            if val and not os.environ.get(key):
                os.environ[key] = val
    except Exception:
        pass

_load_streamlit_secrets()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar — configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    # Pre-fill from env / Streamlit secrets if already set
    _env_key = os.environ.get("OPENAI_API_KEY", "")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=_env_key,
        type="password",
        placeholder="sk-… (leave blank for mock mode)",
        help="Leave blank to run in offline mock mode. Set via Streamlit Secrets on the cloud.",
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
        help=(
            "arXiv, CrossRef, and Europe PMC are 100% free — no key needed.\n"
            "Semantic Scholar requires a free API key to avoid rate limits."
        ),
    )

    generate_hyp = st.toggle("Generate hypotheses", value=True)
    report_fmt   = st.radio("Export format", ["markdown", "json"], horizontal=True)

    st.divider()
    st.caption(
        "📖 **Mock mode** — runs without an API key using placeholder responses. "
        "Useful for testing the full pipeline offline."
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔬 Research Agent")
st.caption("AI-powered literature search · summarisation · hypothesis generation · report writing")

query = st.text_input(
    "Enter your research question or topic",
    placeholder="e.g.  transformer models for protein structure prediction",
)

run_btn = st.button("🚀 Run Research", type="primary", use_container_width=True)

# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn and query.strip():

    # Write key into env — config properties read os.environ live, so this
    # takes effect immediately for all modules in this rerun.
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    elif "OPENAI_API_KEY" in os.environ and not openai_key:
        # User cleared the key field — remove it so mock mode activates
        os.environ.pop("OPENAI_API_KEY", None)

    # Lazy import so the app starts quickly
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

    tick(10, "🔍 Parsing query & extracting keywords…")

    with st.spinner("Running full pipeline — this may take 20–60 s…"):
        try:
            tick(20, "🌐 Searching arXiv and Semantic Scholar…")
            report = agent.run(
                raw_query=query.strip(),
                max_results=int(max_results),
                generate_hypotheses=generate_hyp,
                save_report=False,
            )
            tick(100, "✅ Done!")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    progress.empty()
    status.empty()

    # ── Results ───────────────────────────────────────────────────────────────
    st.success(
        f"Found **{len(report.papers)} papers** · "
        f"Generated **{len(report.hypotheses)} hypotheses**"
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
        st.subheader("Papers Retrieved")
        for i, p in enumerate(report.papers, 1):
            with st.expander(f"{i}. {p.title} ({p.year or 'n.d.'})"):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.caption(f"**Authors:** {', '.join(a.name for a in p.authors[:5])}")
                    st.caption(f"**Venue:** {p.venue or '—'}  |  **Citations:** {p.citation_count}  |  **Source:** {p.source}")
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
                    conf_color = "green" if h.confidence >= 0.6 else "orange" if h.confidence >= 0.4 else "red"
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

elif run_btn and not query.strip():
    st.warning("Please enter a research question first.")

else:
    # Landing state
    st.info(
        "👆 Enter a research question above and click **Run Research** to start.\n\n"
        "**Example queries:**\n"
        "- *transformer models for protein structure prediction*\n"
        "- *CRISPR off-target effects safety 2022-2024*\n"
        "- *quantum error correction surface codes*\n"
        "- *large language model reasoning capabilities*"
    )
