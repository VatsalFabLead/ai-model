#!/usr/bin/env bash
# Hostinger VPS production start (free: gunicorn + uvicorn workers)
set -euo pipefail
cd "$(dirname "$0")/.."
export APP_ENV=production
exec gunicorn app.main:app -c gunicorn.conf.py
