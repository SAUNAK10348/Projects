#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export FUSIONNET_API_TOKEN="${FUSIONNET_API_TOKEN:-}"
uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}"
