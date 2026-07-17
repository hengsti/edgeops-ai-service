FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

RUN useradd --uid 10001 --create-home edgeops

USER edgeops

EXPOSE 8080

CMD ["uvicorn", "edgeops_ai.main:app", "--host", "0.0.0.0", "--port", "8080"]