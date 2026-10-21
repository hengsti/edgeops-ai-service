FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=docker.io/astral/uv:0.13.0 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --frozen --no-dev --extra ml --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra ml --no-editable

COPY artifacts/model.onnx ./artifacts/model.onnx

RUN useradd --uid 10001 --create-home edgeops
USER edgeops

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=3)"]

CMD ["uv","run","--no-sync","uvicorn","edgeops_ai.main:create_app","--factory","--host","0.0.0.0","--port","8080"]
