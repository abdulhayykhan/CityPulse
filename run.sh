#!/usr/bin/env bash
# CityPulse - start the full-stack app (macOS / Linux)
# Usage:  ./run.sh            (defaults to port 8000)
#         ./run.sh 8001
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8000}"

if [ "${CONDA_DEFAULT_ENV:-}" != "citypulse" ]; then
  echo "Tip: activate the env first ->  conda activate citypulse"
fi

echo "Starting CityPulse on http://localhost:${PORT}  (docs at /docs)"
exec uvicorn backend.main:app --reload --port "${PORT}"
