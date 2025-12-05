#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "Virtual environment not found. Run scripts/setup_workspace.sh first." >&2
  exit 1
fi

source .venv/bin/activate
python -m src.realtime "$@"
