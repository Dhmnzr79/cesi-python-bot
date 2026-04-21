#!/usr/bin/env sh
set -eu

# SQLite session storage is single-writer oriented.
# Keep one worker to avoid cross-worker session inconsistencies.
exec gunicorn -w 1 -b 0.0.0.0:8000 app:app
