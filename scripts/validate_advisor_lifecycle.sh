#!/usr/bin/env bash
set -eu

validation_dir=$(cd -- "$(dirname -- "$0")/.." && pwd)
docker run --rm -i --network container:edgeops-ai-service \
  --read-only --no-healthcheck \
  --mount "type=bind,src=$validation_dir/artifacts,dst=/app/artifacts,readonly" \
  --env-file ./edgeops-ai/service/.env \
  -e ADVISOR_INFLUX_URL=http://influxdb:8086/ \
  -e COLLECTOR_POLLING_ENABLED=false -e PREDICTION_BACKEND=rules \
  -e PYTHONPATH=/validation/src -e PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$validation_dir,dst=/validation,readonly" \
  --entrypoint /app/.venv/bin/python \
  "$(docker inspect --format '{{.Image}}' edgeops-ai-service)" \
  /validation/scripts/validate_advisor_lifecycle.py
