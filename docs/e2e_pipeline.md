# End-to-End Pipeline — Complete Guide

This document covers the full E2E pipeline: from trained YOLO weights to final
counting accuracy, including all pre-computed results and how to run new ablations.

---

## Pipeline Architecture

```
SawitMVC-YOLO/images/
        │
        ▼
 ┌─────────────┐
 │ YOLO Detect │  models/y26{n,s,m}_*.pt
 └──────┬──────┘
        │  predictions/{name}_inference/*.json
        ▼
 ┌──────────────────┐
 │ Feature Extract  │  13-dim per tree:
 │                  │  naive_sum, max_per_side,
 │                  │  mean_per_side, n_sides
 └──────┬───────────┘
        │
   ┌────┴────┐────────────┐
   ▼         ▼            ▼
 SVM        RF          M01
 counter  counter   heuristic
   │         │            │
   └────┬────┘────────────┘
        │
        ▼
 benchmarks/e2e/{combo}/metrics.json
```

---

## Complete E2E Results (15 Combinations)

Evaluated on the **test split** (95 trees). Sorted by Acc±1 descending.

| Rank | Detector | Counter | Acc±1 ↑ | MAE ↓ | Total MAE | B1 | B2 | B3 | B4 |
|:----:|----------|---------|:-------:|:-----:|:---------:|:--:|:--:|:--:|:--:|
| 🥇 1 | y26m_vanilla | **SVM** | **71.6%** | **1.118** | 2.432 | 92.6% | 63.2% | 60.0% | 70.5% |
| 2 | y26s_noaug | SVM | 70.5% | 1.126 | 2.232 | 91.6% | 69.5% | 56.8% | 64.2% |
| 3 | y26n_vanilla | SVM | 70.0% | 1.145 | 2.137 | 90.5% | 68.4% | 56.8% | 64.2% |
| 4 | y26s_nopretrained | M01 | 69.2% | 1.266 | 2.874 | 91.6% | 63.2% | 52.6% | 69.5% |
| 5 | y26s_nopretrained | SVM | 69.0% | 1.145 | 2.137 | 90.5% | 68.4% | 51.6% | 65.3% |
| 6 | y26s_vanilla | SVM | 69.0% | 1.163 | 2.337 | 93.7% | 68.4% | 48.4% | 65.3% |
| 7 | y26n_vanilla | RF | 68.2% | 1.218 | 2.242 | 90.5% | 68.4% | 54.7% | 58.9% |
| 8 | y26s_noaug | RF | 68.4% | 1.184 | 2.232 | 92.6% | 66.3% | 55.8% | 58.9% |
| 9 | y26m_vanilla | RF | 67.9% | 1.211 | 2.421 | 95.8% | 68.4% | 49.5% | 57.9% |
| 10 | y26s_nopretrained | RF | 67.9% | 1.229 | 2.284 | 93.7% | 65.3% | 55.8% | 56.8% |
| 11 | y26n_vanilla | M01 | 67.1% | 1.337 | 3.011 | 87.4% | 65.3% | 51.6% | 64.2% |
| 12 | y26s_noaug | M01 | 66.6% | 1.384 | 2.968 | 90.5% | 68.4% | 43.2% | 64.2% |
| 13 | y26s_vanilla | RF | 66.6% | 1.216 | 2.337 | 96.8% | 68.4% | 48.4% | 52.6% |
| 14 | y26s_vanilla | M01 | 65.5% | 1.403 | 3.526 | 89.5% | 66.3% | 38.9% | 67.4% |
| 15 | y26m_vanilla | M01 | 64.5% | 1.400 | 3.853 | 90.5% | 56.8% | 40.0% | 70.5% |
| — | **Heuristic M01 (GT input)** | — | **87.6%** | **0.375** | 1.331 | — | — | — | — |

> Full metrics per combination in `benchmarks/e2e/e2e_{detector}_{counter}/metrics.json`

---

## Key Findings from E2E Results

### 1. SVM consistently beats RF and M01 as a counter

Across all 5 detectors, SVM counter produces the highest Acc±1. RF is second, M01
heuristic is worst when applied to YOLO predictions (not GT). This is because the
M01 heuristic was designed assuming clean GT-quality detections.

### 2. All 15 combinations cluster in 64–72% Acc±1

The narrow spread (~8 pp range) confirms the bottleneck is **detector quality**, not
counter choice. Switching from M01 to SVM gains at most ~7 pp, while improving the
detector from y26s_noaug (0.465 mAP50) to y26m_vanilla (0.509 mAP50) gains ~1 pp.

### 3. B3 is the hardest class in every combination (38–60% Acc±1)

The B2↔B3 visual ambiguity that limits heuristics also limits E2E — YOLO frequently
confuses B2 and B3 labels, compounding the counting error.

### 4. No-augmentation model (y26s_noaug) surprisingly competitive

Despite lowest mAP50 (0.465), y26s_noaug + SVM ranks 2nd overall (70.5%). This
suggests that mAP50 on the test split does not perfectly predict E2E counting accuracy.
The specific error patterns of a detector matter as much as its overall mAP.

### 5. y26m (largest) not always best E2E

y26m_vanilla + SVM = 71.6% (best), but y26m_vanilla + M01 = 64.5% (worst). The
medium model's detection pattern is well-matched to the SVM features but poorly
matched to the M01 heuristic's geometric assumptions.

---

## Running a New Ablation

### Scenario A: Test your own YOLO model

```bash
# 1. Train your model (see docs/training.md)
yolo detect train model=yolo26n.pt data=SawitMVC-YOLO/data.yaml \
    imgsz=640 batch=16 epochs=100 patience=50 seed=42 name=my_experiment

# 2. Run full E2E pipeline
python pipeline/run_e2e_pipeline.py \
    --name my_experiment \
    --weights runs/detect/my_experiment/weights/best.pt

# 3. Compare results
python -c "
import json, os
combos = ['my_experiment_svm', 'my_experiment_rf', 'my_experiment_m01']
for c in combos:
    path = f'benchmarks/e2e/e2e_{c}/metrics.json'
    if os.path.exists(path):
        d = json.load(open(path))['test']
        print(f'{c}: Acc±1={d[\"macro_acc_pm1\"]*100:.1f}%, MAE={d[\"macro_class_mae\"]:.3f}')
"
```

### Scenario B: Test a new counting algorithm (no re-inference)

```bash
# Use any of the 5 pre-computed prediction sets
python pipeline/run_counting_svm.py \
    --inference-dir predictions/y26n_vanilla_local_inference/
# → benchmarks/e2e/e2e_y26n_vanilla_local_svm/metrics.json
```

### Scenario C: New feature engineering

```bash
# Edit pipeline/build_counting_features.py to add your features
# Then re-run with existing predictions:
python pipeline/run_counting_svm.py \
    --inference-dir predictions/y26m_vanilla_local_inference/
```

### Scenario D: Replicate the exact baseline numbers

```bash
# All 5 detectors × 3 counters — uses pre-computed predictions, no GPU needed
for model in y26n_vanilla_local y26s_vanilla_local y26m_vanilla_local y26s_nopretrained y26s_noaug; do
    python pipeline/run_counting_svm.py --inference-dir predictions/${model}_inference/
    python pipeline/run_counting_rf.py  --inference-dir predictions/${model}_inference/
done
```

---

## Metrics Files Structure

Each `benchmarks/e2e/e2e_{detector}_{counter}/` folder contains:

```
metrics.json        ← Primary results (test + val split)
predictions.csv     ← Per-tree predictions vs GT
feature_importance.csv  ← RF only: feature importance ranking
```

`metrics.json` schema (test split):

```json
{
  "test": {
    "macro_acc_pm1": 0.716,
    "macro_class_mae": 1.118,
    "total_count_mae": 2.432,
    "total_pm1_acc": 0.347,
    "exact_profile_acc": 0.021,
    "acc_pm1_B1": 0.926, "acc_pm1_B2": 0.632,
    "acc_pm1_B3": 0.600, "acc_pm1_B4": 0.705,
    "MAE_B1": 0.432, "MAE_B2": 1.242,
    "MAE_B3": 1.600, "MAE_B4": 1.200,
    "bias_B1": -0.074, "bias_B2": -0.653,
    "bias_B3": -0.211, "bias_B4": 0.000,
    "best_params": "{'C': 1, 'gamma': 0.01}"
  },
  "val": { ... }
}
```

---

## Comparison: GT-Upper-Bound vs E2E vs Heuristic

| Setup | Input | Acc±1 | MAE | Gap to ceiling |
|-------|-------|:-----:|:---:|:--------------:|
| SVM on GT features | Perfect detections | **96.1%** | 0.318 | — |
| Heuristic M01 on GT | Perfect detections | 87.6% | 0.375 | −8.5 pp |
| Best E2E (y26m→SVM) | YOLO predictions | 71.6% | 1.118 | −24.5 pp |
| Worst E2E (y26m→M01) | YOLO predictions | 64.5% | 1.400 | −31.6 pp |

The gap between GT-SVM (96.1%) and best E2E (71.6%) = **24.5 pp lost to detector errors**.
This is the opportunity space for future detector improvements.
