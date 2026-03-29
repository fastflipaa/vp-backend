#!/bin/bash
# Entrypoint that routes to the correct service based on SERVICE_ROLE env var
# Each Coolify resource sets SERVICE_ROLE to: api, worker, beat, or flower

set -e

case "${SERVICE_ROLE}" in
  worker)
    exec celery -A app.celery_app worker --loglevel=info --without-heartbeat --without-gossip --without-mingle -Ofair
    ;;
  beat)
    exec celery -A app.celery_app beat --loglevel=info
    ;;
  flower)
    exec celery -A app.celery_app flower --port=5555
    ;;
  api|*)
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --proxy-headers
    ;;
esac
