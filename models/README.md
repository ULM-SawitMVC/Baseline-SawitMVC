# Models

This folder contains the current model artifacts used by the latest baseline.

| Subfolder | Contents |
|-----------|----------|
| [`yolo/`](yolo/) | Current `y26mv2` detector weights and training log. |
| [`counters/`](counters/) | Reusable scikit-learn counter artifacts (SVM, RF, LR). |

## Current detector

The active detector is:

| File | Role |
|------|------|
| [`yolo/y26mv2.pt`](yolo/y26mv2.pt) | Canonical YOLO26m weight used by the latest baseline. |
| [`yolo/y26me60p60b32s42v2.pt`](yolo/y26me60p60b32s42v2.pt) | Original source weight before it is copied to `y26mv2.pt` in the reproduction flow. |
| [`yolo/train_logs/y26m_e60_p60_b32_s42_v2.txt`](yolo/train_logs/y26m_e60_p60_b32_s42_v2.txt) | Training log for the published `y26mv2` run. |

The current README headline numbers are all based on `y26mv2`.

## Reproduction

```bash
cp models/yolo/y26me60p60b32s42v2.pt models/yolo/y26mv2.pt
python pipeline/run_e2e_pipeline.py --name y26mv2 --weights models/yolo/y26mv2.pt
```

## Archived detector families

Older `y26n`, `y26s`, and `y26m` detector weights are archived under
[`archive/models/yolo/`](../archive/models/yolo/). They are historical
experiments and are not part of the latest baseline version.
