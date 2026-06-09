#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python benchmark/benchmark.py \
  --benchmark single \
  --solver team00.py \
  --problem team00_problem.json \
  --mode http \
  --max-games 1 \
  --seeds 0
