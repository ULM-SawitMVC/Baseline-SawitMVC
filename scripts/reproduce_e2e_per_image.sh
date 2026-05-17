#!/usr/bin/env bash
# Reproduce Track B' — per-image end-to-end for every detector x counter.
# Uses cached per-image (or derives them from per-tree) YOLO predictions.
# Writes 21 result folders under results/e2e_per_image/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for detector in y26n y26s y26m; do
    python pipeline/run_e2e_per_image.py --name "$detector" --skip-inference
done
