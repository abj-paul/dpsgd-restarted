#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPIC="${NTFY_TOPIC:-wrn16-1-sketch-progress-20260921-6b2b63b8}"

cd "$PROJECT_DIR"
exec .venv/bin/python -u adaptive_clipping_predictor.py \
  --ntfy-topic "$TOPIC" \
  "$@"
