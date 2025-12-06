#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  echo "Virtualenv not found. Run scripts/setup_workspace.sh first." >&2
  exit 1
fi
source .venv/bin/activate

streamlit run src/dashboard.py --server.headless true
