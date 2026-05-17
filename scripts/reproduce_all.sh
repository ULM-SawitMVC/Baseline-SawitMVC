#!/usr/bin/env bash
# One-shot reproduction of every published number, from cached predictions only.
# Sequence:
#   1. Track A  - five heuristics on all 953 trees.
#   2. Track B  - per-tree end-to-end (3 detectors x 4 counters = 12 results).
#   3. Track B' - per-image end-to-end (3 detectors x 7 counters = 21 results).
#   4. Track C  - ground-truth upper bound (3 counters).
#   5. Verify   - benchmarks/check_release_claims.py exits 0 when the headline
#                 numbers match the README tables.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/5] Track A: heuristic deduplication on 953 trees"
bash scripts/reproduce_heuristics.sh

echo ""
echo "[2/5] Track B: per-tree end-to-end"
bash scripts/reproduce_e2e_per_tree.sh

echo ""
echo "[3/5] Track B': per-image end-to-end"
bash scripts/reproduce_e2e_per_image.sh

echo ""
echo "[4/5] Track C: ground-truth upper bound"
bash scripts/reproduce_upper_bound.sh

echo ""
echo "[5/5] Verifying headline numbers"
python benchmarks/check_release_claims.py
