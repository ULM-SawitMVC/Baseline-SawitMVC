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

## Complete E2E Results (12 Combinations)

Evaluated on the **test split** (95 trees, `split_manifest.csv`). Sorted by Acc±1 descending.

| Rank | Detector | Counter | Acc±1 ↑ | MAE ↓ | B1 | B2 | B3 | B4 |
|:----:|----------|---------|:-------:|:-----:|:--:|:--:|:--:|:--:|
| 🥇 1 | y26n | **M01** | **74.4%** | **1.095** | 95.2% | 78.3% | 54.8% | 69.3% |
| 2 | y26m | M01 | 71.8% | 1.175 | 94.0% | 72.9% | 57.2% | 63.3% |
| 3 | y26s | M01 | 70.9% | 1.224 | 94.0% | 75.9% | 51.8% | 62.0% |
| 4 | y26s | SVM | 70.8% | 1.147 | 93.7% | 66.3% | 53.7% | 69.5% |
| 5 | y26n | SVM | 68.9% | 1.168 | 91.6% | 68.4% | 54.7% | 61.1% |
| 5 | y26m | SVM | 68.9% | 1.168 | 93.7% | 70.5% | 50.5% | 61.1% |
| 7 | y26s | LR | 68.7% | 1.161 | 92.6% | 67.4% | 54.7% | 60.0% |
| 8 | y26n | LR | 68.2% | 1.171 | 92.6% | 72.6% | 53.7% | 53.7% |
| 9 | y26m | LR | 67.9% | 1.174 | 92.6% | 70.5% | 51.6% | 56.8% |
| 10 | y26n | RF | 66.8% | 1.184 | 91.6% | 68.4% | 49.5% | 57.9% |
| 10 | y26m | RF | 66.8% | 1.216 | 90.5% | 64.2% | 54.7% | 57.9% |
| 12 | y26s | RF | 64.2% | 1.255 | 93.7% | 62.1% | 47.4% | 53.7% |
| — | **Heuristic M01 (GT input)** | — | **87.6%** | **0.375** | — | — | — | — |

> Full metrics per combination in `benchmarks/e2e/e2e_{detector}_{counter}/metrics.json`

---

## Key Findings from E2E Results

### 1. M01 heuristic beats ML counters on correctly-trained models

With the corrected `SawitMVC-YOLO` dataset, M01 produces the best Acc±1 across all 3 detectors.
This reverses the finding from the initial (corrupt-dataset) run, where SVM was consistently best.
The implication: the M01 heuristic's geometric assumptions align well with the real dataset.

### 2. All 12 combinations cluster in 64–74% Acc±1

The spread (~10 pp) confirms the bottleneck is **detector quality**. The 74.4% ceiling
(y26n + M01) is ~13 pp below the heuristic-on-GT ceiling (87.6%).

### 3. B3 is the hardest class in every combination (47–57% Acc±1)

The B2↔B3 visual ambiguity limits both YOLO detection and counting. B1 (ripe red bunches)
is consistently the easiest (91–95% Acc±1).

### 4. y26n (smallest model) achieves the best E2E accuracy

Despite having the second-best mAP50 (0.515 vs y26m's 0.528), y26n + M01 achieves
the best overall Acc±1 at 74.4%. Detector error patterns matter as much as mAP50.

### 5. Comparison: GT-Upper-Bound vs E2E

| Setup | Input | Acc±1 | MAE | Gap to ceiling |
|-------|-------|:-----:|:---:|:--------------:|
| Heuristic M01 on GT | Perfect detections | 87.6% | 0.375 | — |
| Best E2E (y26n→M01) | YOLO predictions | 74.4% | 1.095 | −13.2 pp |
| Worst E2E (y26s→RF) | YOLO predictions | 64.2% | 1.255 | −23.4 pp |

The gap from GT-M01 (87.6%) to best E2E (74.4%) = **13.2 pp lost to detector errors**.
Improving detection quality is the highest-leverage improvement.
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
# Use any of the 3 pre-computed prediction sets
python pipeline/run_counting_svm.py \
    --inference-dir predictions/y26n_inference/
# → benchmarks/e2e/e2e_y26n_svm/metrics.json
```


### Scenario C: New feature engineering

```bash
# Edit pipeline/build_counting_features.py to add your features
# Then re-run with existing predictions:
python pipeline/run_counting_svm.py \
    --inference-dir predictions/y26m_inference/
```


### Scenario D: Replicate the exact baseline numbers

```bash
# All 3 detectors × 4 counters — uses pre-computed predictions, no GPU needed
for model in y26n y26s y26m; do
    python pipeline/run_counting_svm.py --inference-dir predictions/${model}_inference/
    python pipeline/run_counting_rf.py  --inference-dir predictions/${model}_inference/
    python pipeline/run_counting_lr.py  --inference-dir predictions/${model}_inference/
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
| Best E2E (y26n→M01) | YOLO predictions | 74.4% | 1.095 | −13.2 pp |
| Worst E2E (y26s→RF) | YOLO predictions | 64.2% | 1.255 | −23.4 pp |

The gap between GT-SVM (96.1%) and best E2E (74.4%) = **21.7 pp lost to detector errors**.
This is the opportunity space for future detector improvements.