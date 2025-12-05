#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <checkpoint.pt> <data_dir> [--device cpu|cuda] [--iterations 20] [--quantize]" >&2
  exit 1
fi

CHECKPOINT=$1
DATA_DIR=$2
shift 2

python -m src.benchmark "$CHECKPOINT" "$DATA_DIR" "$@"
