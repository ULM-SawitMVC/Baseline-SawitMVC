# Changelog

All notable changes to SawitMVC Baseline will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.0] — 2026-05-16

### Changed

**Models replaced — all previous model weights superseded**
- Previous 5 models (`y26n_vanilla_local`, `y26s_vanilla_local`, `y26m_vanilla_local`,
  `y26s_nopretrained`, `y26s_noaug`) were trained on a corrupt dataset and have been removed
- Replaced with 3 correctly-trained models on the validated `SawitMVC-YOLO` dataset:
  - `y26n.pt` — YOLO26n, mAP50=0.515, 5.2 MB, 0.3 ms/frame
  - `y26s.pt` — YOLO26s, mAP50=0.511, 20 MB, 0.4 ms/frame
  - `y26m.pt` — YOLO26m, mAP50=0.528, 42 MB, 1.0 ms/frame
  - All trained: 60 epochs, seed=42, pretrained=True, standard augmentation

**All E2E benchmarks rerun with new models**
- Per-tree: 12 combinations (3 models × 4 counters); best: y26n + M01 = **74.4%**
- Per-image: 21 combinations (3 models × 7 counters); best: y26s + SVM = **70.8%**
- `predictions/` updated: 2,859 per-tree JSONs + 11,976 per-image JSONs
- `benchmarks/e2e/` updated: 33 result folders

---

## [1.1.0] — 2026-05-16

### Added

**E2E Pipeline (complete)**
- `pipeline/` folder with 5 runnable scripts for full E2E replication
- `pipeline/README.md` — step-by-step guide + ablation recipes
- Per-image pipeline extended with ML counters (max/mean/sum/M01/SVM/RF/LR)
- `docs/e2e_pipeline.md` — complete guide with results table, key findings, and ablation scenarios

**Documentation**
- Complete E2E results table in README (all combinations, per-class breakdown)
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

**Models (YOLO26, initial training — superseded in v1.2.0)**
- Initial 5-model set (see v1.2.0 for corrected versions)

**Benchmarks**
- Pre-computed results for all 29 heuristic methods on 953 trees (`accuracy_953.csv`)
- Per-tree predictions CSV (953 rows × 29 methods)
- End-to-end best result: y26m → SVM, Acc±1=71.6% (superseded by v1.2.0)
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
