# images/agent-host/Dockerfile
# The agent-host image: an in-memory LangGraph Agent Server serving the graphs
# in langgraph.json on port 2024. Same image build_image.py produces without
# Docker. MicroVMs need linux/amd64:
#   docker buildx build --platform linux/amd64 -t registry.digitalocean.com/<registry>/microvm-subagent:<tag> --push .
# Runtime configuration (SUBAGENT_HOST_TOKEN, DO_MODEL_ACCESS_KEY, N_JOBS_PER_WORKER)
# arrives as the container's environment, set when the MicroVM is created.
FROM --platform=linux/amd64 python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /opt/agent
COPY pyproject.toml uv.lock ./
RUN uv export --color never --frozen --no-dev --no-emit-project --no-hashes -o /tmp/requirements.txt \
 && uv pip install --system --no-cache -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt
COPY agent.py artifacts_app.py auth.py langgraph.json ./
RUN mkdir -p /work && chown -R 1000:1000 /opt/agent /work

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/work SUBAGENT_WORKDIR=/work PORT=2024
USER 1000:1000
EXPOSE 2024
ENTRYPOINT ["sh", "-c", "exec python -m langgraph_cli dev --config /opt/agent/langgraph.json --host 0.0.0.0 --port \"${PORT:-2024}\" --no-browser --no-reload --allow-blocking --n-jobs-per-worker \"${N_JOBS_PER_WORKER:-2}\""]
