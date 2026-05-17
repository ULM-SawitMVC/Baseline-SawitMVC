# Contributing to SawitMVC Baseline

Thank you for your interest in contributing! This project welcomes contributions in
several areas: new deduplication algorithms, bug fixes, documentation improvements,
and evaluation tooling.

---

## Table of Contents

- [Types of Contributions](#types-of-contributions)
- [Submitting a New Algorithm](#submitting-a-new-algorithm)
- [Reporting Bugs](#reporting-bugs)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Naming Convention](#naming-convention)

---

## Types of Contributions

| Type | Description | Branch prefix |
|------|-------------|--------------|
| New algorithm | A new dedup heuristic with benchmark results | `algo/` |
| Bug fix | Fix incorrect results, broken scripts | `fix/` |
| Documentation | Improve README, docs/, or inline docstrings | `docs/` |
| Benchmark tooling | New evaluation scripts or metrics | `bench/` |

---

## Submitting a New Algorithm

### Requirements (all must be met for inclusion)

- [ ] **Deterministic** — same input always produces same output; no random seeds
- [ ] **No training** — no gradient computation, no embeddings, no learned parameters
- [ ] **Correct signature** — `predict(detections: list) -> dict` exactly
- [ ] **Benchmark result** — run `python benchmarks/run_benchmark.py` and include results
- [ ] **Minimum performance** — Acc±1 ≥ 85.00% on the full 953-tree dataset
- [ ] **Docstring** — includes benchmark result, approach description, and generation note
- [ ] **Self-contained or explicit imports** — only imports from `algorithms/` or stdlib/numpy

### Algorithm file template

```python
"""
M{NN}_{family}_{descriptor}
===========================
Family: {family}
Benchmark 953-tree (SawitMVC-YOLO, post-GT-fix 2026-05-16):
    Acc ±1 = XX.XX%  |  Macro MAE = X.XXXX  |  Total-count MAE = X.XXXX

[2-3 sentence description of the approach]

Constraint: 100% deterministic, no training, no embeddings.
"""

NAMES = ["B1", "B2", "B3", "B4"]


def predict(detections: list) -> dict:
    """
    Predict unique bunch count per maturity class.

    Args:
        detections: List of detection dicts, each with keys:
            - "class"      : str   — "B1", "B2", "B3", or "B4"
            - "x_norm"     : float — normalized x-center of bbox (0.0–1.0)
            - "y_norm"     : float — normalized y-center of bbox (0.0–1.0)
            - "side_index" : int   — camera side index (0-based)

    Returns:
        dict with keys "B1", "B2", "B3", "B4" — predicted unique bunch counts.
    """
    # Your implementation here
    raise NotImplementedError
```

### Steps to submit an algorithm

1. **Fork** the repository and create a branch:
   ```bash
   git checkout -b algo/my-algorithm-name
   ```

2. **Create** your algorithm file at `algorithms/M{NN}_{family}_{descriptor}.py`
   using the template above. Pick the next available NN (check existing files).

3. **Run the benchmark** to get official numbers:
   ```bash
   python benchmarks/run_benchmark.py --save
   ```
   The default `--data` is `./ground_truth/annotations/`, the bundled GT.
   Add the printed Acc±1 / MAE numbers to your docstring.

4. **Register** your algorithm in `algorithms/__init__.py`:
   ```python
   from .M{NN}_your_algorithm import predict as predict_M{NN}

   RANKING["M{NN}_your_algorithm"] = {
       "rank": None,  # filled in after benchmark
       "acc1": 0.XXXX,
       "macro_mae": X.XXXX,
       ...
       "predict": predict_M{NN},
       "description": "...",
   }
   ```

5. **Open a Pull Request** using the PR template. Include the benchmark output
   in the PR description.

---

## Reporting Bugs

Use the [Bug Report issue template](.github/ISSUE_TEMPLATE/bug_report.md).

Please include:
- Python version and OS
- Exact command you ran
- Expected vs actual output
- Minimal reproduction case if possible

---

## Code Style

- **PEP 8** — 4-space indentation, ≤ 88 characters per line
- **Type hints** — required on `predict()`, optional elsewhere
- **No external dependencies** beyond `numpy`, `scipy`, and Python stdlib
  (avoid `pandas`, `torch`, `sklearn` inside algorithm files)
- **No print statements** inside algorithm files; use return values only

---

## Pull Request Process

1. Ensure your algorithm passes the benchmark with Acc±1 ≥ 85.00%
2. Fill in the PR template checklist completely
3. Keep PRs focused — one algorithm or fix per PR
4. PRs are reviewed for correctness (determinism, no training, signature) and
   benchmark reproducibility

Branch naming:
```
algo/m30-entropy-ratio        # new algorithm
fix/m03-floor-off-by-one      # bug fix
docs/evaluation-clarify       # documentation
bench/per-class-breakdown     # tooling
```

---

## Naming Convention

Algorithm files follow the pattern: `M{NN}_{family}_{descriptor}.py`

| Component | Rules |
|-----------|-------|
| `M{NN}` | Two-digit number, continuing from last registered (e.g., M30) |
| `{family}` | One of: `selector`, `blend`, `weight`, `divide`, `median`, `entropy`, `stack`, `anchor`, `consensus` |
| `{descriptor}` | Short snake_case description of the variant |

Examples: `M30_selector_b1b4.py`, `M31_blend_entropy.py`, `M32_weight_aspect.py`

IDs are **permanent** — once assigned, they never change, even if the algorithm is
superseded by a better one.
