#!/usr/bin/env bash
# Reproduce Track B — per-tree end-to-end for y26mv2.
# Uses cached YOLO predictions in predictions/y26mv2_per_tree/.
# Writes result folders under results/e2e_per_tree/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for detector in y26mv2; do
    python pipeline/run_e2e_pipeline.py --name "$detector" --skip-inference
done
