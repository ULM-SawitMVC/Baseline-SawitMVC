#!/usr/bin/env bash
# Archived Track B' experiment.
# Uses cached per-image predictions (or derives them from per-tree) and writes
# archived result folders under archive/results/e2e_per_image/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
for detector in y26mv2; do
    python archive/pipeline/run_e2e_per_image.py --name "$detector" --skip-inference
done
