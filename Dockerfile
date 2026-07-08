# ── Research Agent — Docker image ─────────────────────────────────────────────
#
# Builds a single image that can run either:
#   - The CLI:    docker run research-agent "my query"
#   - The API:    docker run -e MODE=api -p 8000:8000 research-agent
#
# Build:
#   docker build -t research-agent .
#
# Run CLI:
#   docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     -v $(pwd)/reports:/app/reports \
#     research-agent "quantum error correction"
#
# Run API:
#   docker run --rm -e MODE=api -e OPENAI_API_KEY=$OPENAI_API_KEY \
#     -p 8000:8000 research-agent

FROM python:3.12-slim

# Security: run as non-root
RUN addgroup --system agent && adduser --system --ingroup agent agent

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir fastapi uvicorn streamlit

# Copy source
COPY --chown=agent:agent . .
RUN pip install --no-cache-dir -e . --no-deps

# Output directory
RUN mkdir -p /app/reports && chown agent:agent /app/reports
VOLUME ["/app/reports"]

USER agent

ENV REPORT_OUTPUT_DIR=/app/reports
ENV PYTHONUNBUFFERED=1

# Entrypoint: CLI by default, API if MODE=api
COPY --chown=agent:agent docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/docker-entrypoint.sh"]
