FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

RUN useradd --uid 10001 --create-home edgeops
USER edgeops

EXPOSE 8080

CMD ["uv","run","--no-sync","uvicorn","edgeops_ai.main:create_app","--factory","--host","0.0.0.0","--port","8080"]
