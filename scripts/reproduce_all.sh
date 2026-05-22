#!/usr/bin/env bash
# One-shot reproduction of every current published number, from cached
# predictions only.
# Sequence:
#   1. Track A  - heuristic benchmark on all 953 trees.
#   2. Track B  - current per-tree end-to-end baseline for y26mv2.
#   3. Track C  - ground-truth upper bound (5 counters).
#   4. Verify   - benchmarks/check_release_claims.py exits 0 when the headline
#                 numbers match the README tables.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/4] Track A: heuristic deduplication on 953 trees"
bash scripts/reproduce_heuristics.sh

echo ""
echo "[2/4] Track B: per-tree end-to-end"
bash scripts/reproduce_e2e_per_tree.sh

echo ""
echo "[3/4] Track C: ground-truth upper bound"
bash scripts/reproduce_upper_bound.sh

echo ""
echo "[4/4] Verifying headline numbers"
python benchmarks/check_release_claims.py
