#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python benchmark/benchmark.py \
  --benchmark bulk \
  --solver team00.py \
  --words words.txt \
  --mode direct \
  --num-problems 3 \
  --candidate-size 500 \
  --include-default-openers \
  --seeds 0
