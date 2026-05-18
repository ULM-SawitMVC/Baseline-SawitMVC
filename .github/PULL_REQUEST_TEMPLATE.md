## Description

<!-- Briefly describe what this PR adds or fixes. -->

## Type of Change

- [ ] New algorithm (`algorithms/M{NN}_*.py`)
- [ ] Bug fix
- [ ] Documentation improvement
- [ ] Benchmark tooling

---

## Checklist

### For New Algorithms

- [ ] Algorithm file created at `algorithms/M{NN}_{family}_{descriptor}.py`
- [ ] `predict(detections: list) -> dict` signature is correct
- [ ] Algorithm is fully deterministic (no random seeds, no training)
- [ ] Tested on the full 953-tree dataset (`python benchmarks/run_benchmark.py`)
- [ ] Acc±1 ≥ 85.00% on 953 trees
- [ ] Benchmark results included in docstring
- [ ] Registered in `algorithms/__init__.py` RANKING dict
- [ ] Named following the `M{NN}_{family}_{descriptor}` convention

### For Bug Fixes

- [ ] Root cause identified and explained in PR description
- [ ] Fix does not change benchmark results for unaffected algorithms
- [ ] Tested with `python benchmarks/run_benchmark.py`

### General

- [ ] Code follows PEP 8
- [ ] No print statements in algorithm files
- [ ] PR is focused, one algorithm or fix per PR

---

## Benchmark Results

<!-- Paste the output of `python benchmarks/run_benchmark.py` here -->

```
SawitMVC Baseline, Benchmark Results (953 trees)
...
```

---

## Additional Notes

<!-- Anything else reviewers should know. -->
