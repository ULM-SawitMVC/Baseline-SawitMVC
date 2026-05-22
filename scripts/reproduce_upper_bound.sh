#!/usr/bin/env bash
# Reproduce Track C — ground-truth upper bound.
# Fits SVM, RF, LR, Ridge, and ElasticNet on features derived from the GT
# annotations themselves (rather than YOLO predictions).
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
python pipeline/run_counting_regularized.py \
    --model ridge \
    --inference-dir ground_truth/annotations \
    --out results/e2e_upper_bound/gt_ridge
python pipeline/run_counting_regularized.py \
    --model elasticnet \
    --inference-dir ground_truth/annotations \
    --out results/e2e_upper_bound/gt_elasticnet
