"""
Command-line interface for the Research Agent.

Usage
-----
  python -m research_agent [OPTIONS] QUERY

  research_agent --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from typing import List


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research_agent",
        description="AI-powered Research Agent — search, summarise, hypothesise, report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              research_agent "transformer models for protein folding"
              research_agent "CRISPR gene editing safety" --results 15 --style ieee
              research_agent "quantum computing error correction" --providers arxiv --no-hypotheses
              research_agent "climate tipping points" --format json --output ./my_reports
            """
        ),
    )

    # ── Positional ────────────────────────────────────────────────────────────
    parser.add_argument(
        "query",
        help="Research question or topic (enclose multi-word queries in quotes).",
    )

    # ── Search options ────────────────────────────────────────────────────────
    parser.add_argument(
        "-n", "--results",
        type=int,
        default=10,
        metavar="N",
        help="Maximum papers to retrieve per search provider (default: 10).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["arxiv", "semantic_scholar"],
        default=None,
        metavar="PROVIDER",
        help="Search providers to use (default: all).",
    )

    # ── Output options ────────────────────────────────────────────────────────
    parser.add_argument(
        "--format",
        dest="report_format",
        choices=["markdown", "json"],
        default="markdown",
        help="Report output format (default: markdown).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="DIR",
        help="Directory to write the report to (default: ./reports).",
    )
    parser.add_argument(
        "--style",
        choices=["apa", "mla", "ieee", "chicago"],
        default="apa",
        help="Citation style (default: apa).",
    )

    # ── Pipeline flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-hypotheses",
        action="store_true",
        help="Skip the hypothesis-generation step.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the report to disk; print to stdout instead.",
    )
    parser.add_argument(
        "--save-citations",
        metavar="FILE",
        default=None,
        help="Save the session citation library to a JSON file.",
    )
    parser.add_argument(
        "--load-citations",
        metavar="FILE",
        default=None,
        help="Pre-load a citation library from a JSON file.",
    )

    # ── Verbosity ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all informational output.",
    )

    return parser


def _setup_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose, args.quiet)

    # Lazy import to avoid circular imports and slow startup for --help
    from research_agent.agent import ResearchAgent
    from research_agent import reporter

    agent = ResearchAgent(
        citation_style=args.style,
        providers=args.providers,
    )

    if args.load_citations:
        try:
            agent.load_citations(args.load_citations)
        except FileNotFoundError:
            print(f"[warning] Citation file not found: {args.load_citations}", file=sys.stderr)

    try:
        report = agent.run(
            raw_query=args.query,
            max_results=args.results,
            generate_hypotheses=not args.no_hypotheses,
            save_report=not args.no_save,
            report_format=args.report_format,
            output_dir=args.output,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if args.no_save:
        if args.report_format == "json":
            print(reporter.render_json(report))
        else:
            print(reporter.render_markdown(report))

    if args.save_citations:
        agent.save_citations(args.save_citations)

    if not args.quiet:
        print(
            f"\n✓ Done — {len(report.papers)} papers | "
            f"{len(report.hypotheses)} hypotheses",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
