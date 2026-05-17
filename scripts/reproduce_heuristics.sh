#!/usr/bin/env bash
# Reproduce Track A — the five deterministic heuristics on the full 953-tree set.
# Reads ground_truth/annotations/, writes results/heuristics_953/benchmark_top5.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python benchmarks/run_benchmark.py --save
