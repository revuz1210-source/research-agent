#!/bin/sh
# Switch between CLI and API mode via the MODE env var.
# MODE=api   → start FastAPI server
# MODE=ui    → start Streamlit app
# (default)  → run CLI with forwarded arguments

set -e

case "${MODE:-cli}" in
  api)
    exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" ;;
  ui)
    exec streamlit run streamlit_app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 ;;
  *)
    exec python -m research_agent "$@" ;;
esac
