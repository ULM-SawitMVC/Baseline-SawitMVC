#!/usr/bin/env bash
# Reproduce Track B — per-tree end-to-end for every detector x counter.
# Uses cached YOLO predictions in predictions/{detector}_per_tree/.
# Writes 12 result folders under results/e2e_per_tree/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for detector in y26n y26s y26m; do
    python pipeline/run_e2e_pipeline.py --name "$detector" --skip-inference
done
