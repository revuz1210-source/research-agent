"""
Central configuration for the Research Agent.
Values are read from environment variables on every access — safe for
long-running servers and Streamlit sessions where env vars may be set
after the module is first imported.

On startup, automatically loads a .env file from the project root if
python-dotenv is installed — so keys saved in .env work immediately
without any extra setup.
"""

import os
from typing import Optional


def _load_dotenv() -> None:
    """Load .env into os.environ if python-dotenv is available."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        import io
        # Read raw bytes and detect encoding (UTF-16 BOM is common on Windows)
        raw = open(env_path, "rb").read()
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8", errors="replace")
        load_dotenv(stream=io.StringIO(text), override=False)
    except ImportError:
        # python-dotenv not installed — fall back to manual parse
        _parse_dotenv_manually()
    except Exception:
        # Any other issue — try manual parser as last resort
        _parse_dotenv_manually()


def _parse_dotenv_manually() -> None:
    """
    Minimal .env parser — no dependencies needed.
    Handles KEY=value and KEY="value" lines, skips comments.
    Detects UTF-8 and UTF-16 (Windows Notepad default) encodings.
    Only sets variables that are not already in the environment.
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    raw = open(env_path, "rb").read()
    # Detect UTF-16 BOM written by Windows Notepad / VS Code "Save As UTF-16"
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


class Config:
    """
    All attributes are dynamic properties — they read from os.environ
    each time they are accessed, so setting os.environ after import
    takes effect immediately everywhere.
    """

    # ── LLM ───────────────────────────────────────────────────────────────────
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4o")

    @property
    def openai_temperature(self) -> float:
        return float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

    @property
    def openai_max_tokens(self) -> int:
        return int(os.getenv("OPENAI_MAX_TOKENS", "2048"))

    # ── Search APIs ───────────────────────────────────────────────────────────
    @property
    def semantic_scholar_api_key(self) -> Optional[str]:
        return os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None

    @property
    def arxiv_max_results(self) -> int:
        return int(os.getenv("ARXIV_MAX_RESULTS", "10"))

    @property
    def semantic_scholar_max_results(self) -> int:
        return int(os.getenv("SEMANTIC_SCHOLAR_MAX_RESULTS", "10"))

    # ── Summarisation ─────────────────────────────────────────────────────────
    @property
    def summary_max_length(self) -> int:
        return int(os.getenv("SUMMARY_MAX_LENGTH", "300"))

    @property
    def summary_style(self) -> str:
        return os.getenv("SUMMARY_STYLE", "academic")

    # ── Reports ───────────────────────────────────────────────────────────────
    @property
    def report_output_dir(self) -> str:
        return os.getenv("REPORT_OUTPUT_DIR", "reports")

    @property
    def citation_style(self) -> str:
        return os.getenv("CITATION_STYLE", "apa")

    # ── Hypothesis ────────────────────────────────────────────────────────────
    @property
    def hypothesis_count(self) -> int:
        return int(os.getenv("HYPOTHESIS_COUNT", "3"))

    # ── Misc ──────────────────────────────────────────────────────────────────
    @property
    def request_timeout(self) -> int:
        return int(os.getenv("REQUEST_TIMEOUT", "30"))


# Shared singleton — properties read env vars live on every access
config = Config()
