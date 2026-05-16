# Changelog

All notable changes to SawitMVC Baseline will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-05-16

### Added

**E2E Pipeline (complete)**
- `pipeline/` folder with 5 runnable scripts for full E2E replication
- `pipeline/README.md` — step-by-step guide + ablation recipes
- `predictions/` — pre-computed YOLO inference for all 5 models (4,765 JSON, ~28 MB)
- All 15 E2E result folders in `benchmarks/e2e/` (5 detectors × 3 counters)
- `docs/e2e_pipeline.md` — complete guide with 15-combination results table, key
  findings, and 4 ablation scenarios with exact commands

**Documentation**
- Complete E2E results table in README (all 15 combinations, per-class breakdown)
- Updated Repository Structure section with `pipeline/` and `predictions/`

---

## [1.0.0] — 2026-05-16

### Added

**Algorithms**
- `M01_selector_b2b3` — Champion heuristic, 87.62% Acc±1 on 953 trees
- `M02_selector_trifurc` — Trifurcation selector (base of M01), 87.62% Acc±1
- `M03_blend_geometric` — Geometric mean blend, 86.99% Acc±1
- `M04_blend_floor_clamped` — Floor-clamped weighted blend, 86.99% Acc±1
- `M05_blend_vis_divide` — Simple weighted blend, 86.99% Acc±1

**Models (YOLO26, trained locally)**
- `y26n_vanilla_local.pt` — Best: mAP50=0.521, 5.2 MB, 0.2 ms/frame
- `y26s_vanilla_local.pt` — mAP50=0.506, 20 MB
- `y26m_vanilla_local.pt` — mAP50=0.509, 42 MB (best E2E pipeline)
- `y26s_nopretrained.pt` — Ablation: scratch training, mAP50=0.511
- `y26s_noaug.pt` — Ablation: no augmentation, mAP50=0.465

**Benchmarks**
- Pre-computed results for all 29 heuristic methods on 953 trees (`accuracy_953.csv`)
- Per-tree predictions CSV (953 rows × 29 methods)
- End-to-end best result: y26m → SVM, Acc±1=71.6%
- Runnable benchmark script (`benchmarks/run_benchmark.py`)

**Figures**
- 23 EDA plots (class distribution, spatial heatmaps, appearance analysis)
- 35 training plots (5 models × 7 types: confusion matrices, PR curves, training curves)

**Documentation**
- `docs/dataset.md` — Dataset description, classes, GT quality
- `docs/algorithms.md` — Algorithm design and rationale
- `docs/training.md` — YOLO training reproduction guide
- `docs/evaluation.md` — Metric definitions and evaluation protocol
- `docs/findings.md` — Key research findings

**Ground truth status (as of this release)**
- 0 same-side uniqueness violations (8 wrap-around trees fixed)
- 0 geometric adjacency violations (31 four-side trees auto-healed, 9 eight-side trees manually fixed)
- 9,823 unique bunches verified across 953 trees
