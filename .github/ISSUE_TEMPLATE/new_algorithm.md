---
name: New Algorithm Proposal
about: Propose a new deduplication algorithm for inclusion in the baseline
title: "[ALGO] "
labels: new-algorithm
assignees: ''
---

## Algorithm Name and Family

**Proposed name:** `M{NN}_{family}_{descriptor}`  
**Family:** (selector / blend / weight / divide / median / entropy / stack / other)

## Brief Description

Describe the core idea in 2–4 sentences. What geometric or statistical insight
does this algorithm exploit? How is it different from the existing top-5?

## Benchmark Results

Run `python benchmarks/run_benchmark.py --save` and paste the results for your
algorithm (the default `--data` is `./ground_truth/annotations/`):

```
Algorithm: M{NN}_your_algorithm
Dataset:   SawitMVC-YOLO (953 trees, post-GT-fix 2026-05-16)

Acc ±1:          XX.XX%
Macro class-MAE: X.XXXX
Total-count MAE: X.XXXX
n_fail:          XXX
```

Per-class breakdown (optional but appreciated):
```
MAE B1: X.XXXX   MAE B2: X.XXXX   MAE B3: X.XXXX   MAE B4: X.XXXX
```

## Algorithm Code (draft)

```python
"""
M{NN}_{family}_{descriptor}
===========================
Family: {family}
Benchmark 953-tree: Acc ±1 = XX.XX%, MAE = X.XXXX

[Brief description]
"""

NAMES = ["B1", "B2", "B3", "B4"]

def predict(detections: list) -> dict:
    """..."""
    # implementation
    pass
```

## Compliance Checklist

- [ ] Fully deterministic (same input → same output, no random seeds)
- [ ] Zero training, embeddings, or gradient computation
- [ ] `predict(detections: list) -> dict` signature exactly as specified
- [ ] Tested on the full 953-tree dataset
- [ ] Acc±1 ≥ 85.00% (minimum bar for inclusion)
- [ ] Docstring includes benchmark result
- [ ] No external dependencies beyond numpy and Python stdlib

## Comparison to Existing Methods

How does this algorithm compare to M01–M05? What specific weakness of the existing
methods does it address?
