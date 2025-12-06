#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RECREATE=false
if [[ ${1-} == "--recreate" ]]; then
  RECREATE=true
  shift
fi

if $RECREATE && [[ -d .venv ]]; then
  rm -rf .venv
fi

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment created. Activate with:\nsource .venv/bin/activate"
