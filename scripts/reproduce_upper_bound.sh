#!/usr/bin/env bash
# Reproduce Track C — ground-truth upper bound.
# Fits SVM, RF, and LR on features derived from the GT annotations themselves
# (rather than YOLO predictions). Writes results/e2e_upper_bound/gt_{svm,rf,lr}/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python pipeline/run_counting_svm.py \
    --inference-dir ground_truth/annotations \
    --out results/e2e_upper_bound/gt_svm
python pipeline/run_counting_rf.py \
    --inference-dir ground_truth/annotations \
    --out results/e2e_upper_bound/gt_rf
python pipeline/run_counting_lr.py \
    --inference-dir ground_truth/annotations \
    --out results/e2e_upper_bound/gt_lr
