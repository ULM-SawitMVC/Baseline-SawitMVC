# SawitMVC Baseline — Multi-View Oil Palm Bunch Counting

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dataset: HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO)

**SawitMVC Baseline** is the official research baseline for multi-view oil palm fruit bunch
counting and maturity classification. Given 4–8 photos of the same oil palm tree taken from
different angles, the task is to count the **unique number of bunches per maturity class**
(B1 → B4) without double-counting bunches visible from multiple sides.

> **Core problem:** Naive summation overcounts by ~83.4% because the same bunch is
> detected across multiple camera angles. This repository provides 5 deterministic
> heuristic algorithms that reduce the error to ≤13% without any training.

> **Note:** All YOLO baselines were retrained on the correct `SawitMVC-YOLO` dataset
> (3 models: `y26n`, `y26s`, `y26m`). Previous 5-model results have been superseded.

---

## Artifact Links

| Category | Clickable files |
|----------|-----------------|
| Model weights | [`models/y26n.pt`](models/y26n.pt), [`models/y26s.pt`](models/y26s.pt), [`models/y26m.pt`](models/y26m.pt) |
| Training logs | [`models/y26n_train_log.txt`](models/y26n_train_log.txt), [`models/y26s_train_log.txt`](models/y26s_train_log.txt), [`models/y26m_train_log.txt`](models/y26m_train_log.txt) |
| Heuristic benchmark | [`benchmarks/results/accuracy_953.csv`](benchmarks/results/accuracy_953.csv), [`benchmarks/results/per_tree.csv`](benchmarks/results/per_tree.csv), [`benchmarks/results/totals.csv`](benchmarks/results/totals.csv), [`benchmarks/results/mean_per_tree.csv`](benchmarks/results/mean_per_tree.csv) |
| E2E metrics | [`benchmarks/e2e/e2e_y26s_svm/metrics.json`](benchmarks/e2e/e2e_y26s_svm/metrics.json), [`benchmarks/e2e/e2e_y26m_m01/metrics.json`](benchmarks/e2e/e2e_y26m_m01/metrics.json), [`benchmarks/e2e/e2e_y26n_svm/metrics.json`](benchmarks/e2e/e2e_y26n_svm/metrics.json) |
| E2E predictions | [`benchmarks/e2e/e2e_y26s_svm/predictions.csv`](benchmarks/e2e/e2e_y26s_svm/predictions.csv), [`benchmarks/e2e/e2e_y26m_m01/predictions.csv`](benchmarks/e2e/e2e_y26m_m01/predictions.csv), [`benchmarks/e2e/e2e_y26n_svm/predictions.csv`](benchmarks/e2e/e2e_y26n_svm/predictions.csv) |
| YOLO predictions | [`predictions/y26n_inference/`](predictions/y26n_inference/), [`predictions/y26s_inference/`](predictions/y26s_inference/), [`predictions/y26m_inference/`](predictions/y26m_inference/) |
| Pipeline scripts | [`pipeline/run_e2e_pipeline.py`](pipeline/run_e2e_pipeline.py), [`pipeline/run_e2e_per_image.py`](pipeline/run_e2e_per_image.py), [`pipeline/build_counting_features.py`](pipeline/build_counting_features.py) |
| Validation script | [`benchmarks/check_release_claims.py`](benchmarks/check_release_claims.py) |

## Results at a Glance

### Heuristic Deduplication (953 trees, no training required)

| Rank | Algorithm | Acc ±1 | Macro MAE | Approach |
|:----:|-----------|:------:|:---------:|----------|
| 🥇 1 | [`M01_selector_b2b3.py`](algorithms/M01_selector_b2b3.py) | **87.62%** | 0.375 | Trifurcation selector + B2↔B3 correction |
| 🥈 2 | [`M02_selector_trifurc.py`](algorithms/M02_selector_trifurc.py) | 87.62% | 0.376 | Trifurcation selector (base) |
| 🥉 3 | [`M03_blend_geometric.py`](algorithms/M03_blend_geometric.py) | 86.99% | 0.377 | Geometric mean blend |
| 4 | [`M04_blend_floor_clamped.py`](algorithms/M04_blend_floor_clamped.py) | 86.99% | 0.385 | Floor-clamped weighted blend |
| 5 | [`M05_blend_vis_divide.py`](algorithms/M05_blend_vis_divide.py) | 86.99% | 0.388 | Visibility + adaptive divide |
| — | Naive sum (baseline) | 3.78% | 2.287 | No deduplication |

**Acc ±1** = percentage of trees where predicted count per class is within ±1 of ground truth (macro-averaged across 4 classes). Evaluated on 953 trees from `Brand-New-Dataset-YOLO`.

### YOLO26 Detection (mAP50, val split)

All 3 models trained 60 epochs, seed=42, pretrained=True, standard augmentation on `SawitMVC-YOLO`.

| Model | mAP50 | Speed | Size | Notes |
|-------|:-----:|:-----:|:----:|-------|
| [`y26m.pt`](models/y26m.pt) | **0.528** | 1.0 ms | 42 MB | Best detection accuracy; log: [`y26m_train_log.txt`](models/y26m_train_log.txt) |
| [`y26n.pt`](models/y26n.pt) | 0.515 | **0.3 ms** | **5.2 MB** | **Best efficiency (recommended)**; log: [`y26n_train_log.txt`](models/y26n_train_log.txt) |
| [`y26s.pt`](models/y26s.pt) | 0.511 | 0.4 ms | 20 MB | Balanced size/speed; log: [`y26s_train_log.txt`](models/y26s_train_log.txt) |

### E2E — Per-Image Approach (complete pipeline, 95 test trees)

Each image is processed independently by YOLO, then grouped per tree. Counting is done
with either simple aggregation (`max`/`mean`/`sum`) or the same ML/heuristic counters
as the per-tree approach. Results on test split (95 trees, `split_manifest.csv`).

> **Key finding:** Per-image + ML counter achieves **identical accuracy** to per-tree + ML counter,
> because the 13-dim features are derived from the same per-image detection counts either way.
> Per-image inference is therefore sufficient — no need to group images before YOLO runs.

```bash
# Run all counters for one model (derives per-image JSONs from existing per-tree predictions)
python pipeline/run_e2e_per_image.py \
    --name y26n --weights models/y26n.pt \
    --data ./SawitMVC-YOLO --skip-inference
```

**Simple aggregation (no training):** 3 models sorted by Acc±1

| Rank | Detector | Agg | Acc ±1 | MAE |
|:----:|----------|-----|:------:|:---:|
| 1 | y26m | **max** | **64.2%** | 1.368 |
| 2 | y26n | max | 63.7% | 1.363 |
| 3 | y26s | max | 61.8% | 1.403 |
| — | *(all models)* | mean | 53.9–55.8% | 1.70–1.84 |
| — | *(all models)* | sum | 50.0–52.6% | 2.22–2.48 |

`max` is the best simple dedup: a bunch visible from multiple sides is counted once (best view).
`sum` is worst: overcounts the same bunch detected from each side.

**With ML/heuristic counter:** per-image achieves the same results as per-tree (see table below).
All 21 per-image results (3 models × 7 counters) in [`benchmarks/e2e/`](benchmarks/e2e/) as
`e2e_{model}_per_image_{counter}/metrics.json`.

---

### E2E — Per-Tree Approach (full pipeline, pre-computed)

All images of a tree are grouped, detections aggregated into 13-dim features, then a ML
counter is applied. Results on test split (95 trees, `split_manifest.csv`).
Sorted by Acc±1. **4 counters × 3 detectors = 12 combinations.**

| Rank | Detector | Counter | Acc ±1 ↑ | MAE ↓ | B1 | B2 | B3 | B4 |
|:----:|----------|---------|:--------:|:-----:|:--:|:--:|:--:|:--:|
| 🥇 1 | [`y26s.pt`](models/y26s.pt) | [SVM](benchmarks/e2e/e2e_y26s_svm/metrics.json) | **70.8%** | 1.147 | 93.7% | 66.3% | 53.7% | 69.5% |
| 2 | [`y26m.pt`](models/y26m.pt) | [M01](benchmarks/e2e/e2e_y26m_m01/metrics.json) | 69.2% | 1.295 | 91.6% | 69.5% | 48.4% | 67.4% |
| 3 | [`y26n.pt`](models/y26n.pt) | [SVM](benchmarks/e2e/e2e_y26n_svm/metrics.json) | 68.9% | 1.168 | 91.6% | 68.4% | 54.7% | 61.1% |
| 4 | [`y26m.pt`](models/y26m.pt) | [SVM](benchmarks/e2e/e2e_y26m_svm/metrics.json) | 68.9% | 1.168 | 93.7% | 70.5% | 50.5% | 61.1% |
| 5 | [`y26s.pt`](models/y26s.pt) | [LR](benchmarks/e2e/e2e_y26s_lr/metrics.json) | 68.7% | 1.161 | 92.6% | 67.4% | 54.7% | 60.0% |
| 6 | [`y26n.pt`](models/y26n.pt) | [LR](benchmarks/e2e/e2e_y26n_lr/metrics.json) | 68.2% | 1.171 | 92.6% | 72.6% | 53.7% | 53.7% |
| 7 | [`y26m.pt`](models/y26m.pt) | [LR](benchmarks/e2e/e2e_y26m_lr/metrics.json) | 67.9% | 1.174 | 92.6% | 70.5% | 51.6% | 56.8% |
| 8 | [`y26m.pt`](models/y26m.pt) | [RF](benchmarks/e2e/e2e_y26m_rf/metrics.json) | 66.8% | 1.216 | 90.5% | 64.2% | 54.7% | 57.9% |
| 9 | [`y26n.pt`](models/y26n.pt) | [RF](benchmarks/e2e/e2e_y26n_rf/metrics.json) | 66.8% | 1.184 | 91.6% | 68.4% | 49.5% | 57.9% |
| 10 | [`y26s.pt`](models/y26s.pt) | [M01](benchmarks/e2e/e2e_y26s_m01/metrics.json) | 64.2% | 1.313 | 89.5% | 55.8% | 49.5% | 62.1% |
| 11 | [`y26s.pt`](models/y26s.pt) | [RF](benchmarks/e2e/e2e_y26s_rf/metrics.json) | 64.2% | 1.255 | 93.7% | 62.1% | 47.4% | 53.7% |
| 12 | [`y26n.pt`](models/y26n.pt) | [M01](benchmarks/e2e/e2e_y26n_m01/metrics.json) | 63.9% | 1.342 | 89.5% | 64.2% | 43.2% | 58.9% |
| — | **M01 on GT** (upper bound) | — | **87.6%** | **0.375** | — | — | — | — |

> Full analysis in [`docs/e2e_pipeline.md`](docs/e2e_pipeline.md).
> All `metrics.json` files in [`benchmarks/e2e/`](benchmarks/e2e/).

The ~17 pp gap between best E2E (70.8%) and heuristic-on-GT represents **detector error
propagation** — improving the detector is the highest-leverage path. See [`docs/findings.md`](docs/findings.md).

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

The dataset is external and is not included in this repository. The precomputed
prediction JSONs are included under `predictions/`, but the ground-truth JSONs from
`SawitMVC-YOLO/json/` are still required to re-run benchmarks or E2E evaluation.
Hugging Face access may require an approved account token for this dataset.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ULM-DS-Lab/SawitMVC-YOLO",
    repo_type="dataset",
    local_dir="./SawitMVC-YOLO",
    token=True,  # uses your logged-in Hugging Face token when access is gated
)
```

### 3. Run the benchmark

```bash
python benchmarks/run_benchmark.py --data ./SawitMVC-YOLO/json/
```

Expected output:

```
SawitMVC Baseline — Benchmark Results (953 trees)
==================================================
Rank  Algorithm               Acc±1    MAE
  1   M01_selector_b2b3      87.62%  0.3746
  2   M02_selector_trifurc   87.62%  0.3757
  3   M03_blend_geometric    86.99%  0.3767
  4   M04_blend_floor_clamped 86.99%  0.3848
  5   M05_blend_vis_divide   86.99%  0.3875
```

### 4. Run inference on a single tree

```python
import json
from algorithms.M01_selector_b2b3 import predict

# Load detections from a JSON ground-truth file (or your own detector output)
with open("SawitMVC-YOLO/json/DAMIMAS_A21B_0001.json") as f:
    tree = json.load(f)

# Build detection list from JSON
detections = []
for side_key, side in tree["images"].items():
    for ann in side["annotations"]:
        x_center = ann["bbox_yolo"][0]
        detections.append({
            "class": ann["class_name"],
            "x_norm": x_center,
            "y_norm": ann["bbox_yolo"][1],
            "side_index": side["side_index"],
        })

result = predict(detections)
print(result)  # {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
```

---

## Repository Structure

```
Baseline-SawitMVC/
├── README.md                    You are here
├── CONTRIBUTING.md              How to contribute new algorithms or fixes
├── CHANGELOG.md                 Version history
├── LICENSE                      CC BY-NC 4.0
├── requirements.txt             Python dependencies
│
├── algorithms/                  Top-5 heuristic deduplication algorithms
│   ├── README.md                Algorithm guide, comparison table, input/output spec
│   ├── __init__.py              Unified imports + ranking metadata
│   ├── M01_selector_b2b3.py    🥇 Champion: 87.62% Acc±1
│   ├── M02_selector_trifurc.py 🥈 Runner-up: 87.62% Acc±1
│   ├── M03_blend_geometric.py  🥉 86.99% Acc±1
│   ├── M04_blend_floor_clamped.py  86.99% Acc±1
│   └── M05_blend_vis_divide.py     86.99% Acc±1
│
├── models/                      Trained YOLO26 weights (all < 50 MB)
│   ├── README.md                Model comparison, inference guide
│   ├── y26n.pt                 ⚡ Best efficiency: mAP50=0.515, 5.2 MB
│   ├── y26s.pt                 Standard small: mAP50=0.511, 20 MB
│   └── y26m.pt                 Best accuracy: mAP50=0.528, 42 MB
│
├── benchmarks/                  Reproducible benchmark suite
│   ├── README.md                How to run, metric definitions
│   ├── run_benchmark.py         Entry point: load data → run → print table
│   ├── results/                 Pre-computed results (953 trees, 29 methods)
│   │   ├── accuracy_953.csv     Full 29-method ranking
│   │   ├── per_tree.csv         Per-tree predictions
│   │   ├── totals.csv           Aggregate counts
│   │   └── mean_per_tree.csv    Mean per-tree statistics
│   └── e2e/                     All 33 E2E results (3 detectors × 11 counters)
│       ├── e2e_y26n_m01/              metrics.json + predictions.csv
│       ├── e2e_y26n_per_image_svm/
│       └── … (33 folders total)
│
├── pipeline/                    E2E pipeline scripts
│   ├── README.md                Step-by-step replication + ablation guide
│   ├── run_e2e_pipeline.py      Unified harness (inference → features → eval)
│   ├── run_e2e_inference.py     YOLO inference → JSON per tree
│   ├── build_counting_features.py  Extract 13-dim features
│   ├── run_counting_svm.py      SVM counter (train + evaluate)
│   └── run_counting_rf.py       RF counter (train + evaluate)
│
├── predictions/                 Pre-computed YOLO inference (~17 MB)
│   ├── y26n_inference/          953 JSON files (per-tree)
│   ├── y26s_inference/          953 JSON files (per-tree)
│   ├── y26m_inference/          953 JSON files (per-tree)
│   ├── y26n_per_image/          3992 JSON files (per-image, derived)
│   ├── y26s_per_image/          3992 JSON files (per-image, derived)
│   └── y26m_per_image/          3992 JSON files (per-image, derived)
│
├── figures/                     All visualizations
│   ├── README.md                Figure index and descriptions
│   ├── eda/                     23 EDA plots (dataset characteristics)
│   └── training/                YOLO training artifacts
│
└── docs/                        In-depth documentation
    ├── dataset.md               Dataset structure, classes, GT quality
    ├── algorithms.md            How algorithms work, design rationale
    ├── training.md              Reproduce YOLO training experiments
    ├── evaluation.md            Metric definitions, evaluation protocol
    ├── e2e_pipeline.md          Full E2E guide + 15-combo results + ablation recipes
    └── findings.md              Key research findings and future work
```

---

Clickable repository paths:
[`README.md`](README.md),
[`CONTRIBUTING.md`](CONTRIBUTING.md),
[`CHANGELOG.md`](CHANGELOG.md),
[`requirements.txt`](requirements.txt),
[`algorithms/`](algorithms/),
[`models/`](models/),
[`benchmarks/`](benchmarks/),
[`benchmarks/results/accuracy_953.csv`](benchmarks/results/accuracy_953.csv),
[`benchmarks/e2e/`](benchmarks/e2e/),
[`pipeline/`](pipeline/),
[`predictions/`](predictions/),
[`figures/`](figures/),
[`docs/`](docs/).

## Dataset

**SawitMVC-YOLO** — 953 oil palm trees from DAMIMAS (854) and LONSUM (99) plantations
in Kalimantan, Indonesia.

| Statistic | Value |
|-----------|-------|
| Trees | 953 |
| Images | 3,992 JPEG (960×1280) |
| Unique bunches | 9,823 |
| Raw detections | 18,544 |
| Deduplication ratio | 0.53 (naive overcounts 1.83×) |
| 4-side trees | 908 (95.3%) |
| 8-side trees | 45 (4.7%) |
| Ground truth format | JSON with confirmed cross-side links |

**Maturity classes (ordinal, most → least mature):**

| Class | Visual | Physical Position |
|-------|--------|-------------------|
| **B1** | Red, large, round | Lowest on tree |
| **B2** | Black with red transition | Above B1 |
| **B3** | Full black, spiky, elongated | Above B2 |
| **B4** | Smallest, dark green-black | Highest (newest) |

Download: [HuggingFace — ULM-DS-Lab/SawitMVC-YOLO](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO)

---

## How to Contribute

We welcome contributions of:

- **New deduplication algorithms** — must be deterministic, no training, better Acc±1 than M05
- **Bug reports** — incorrect benchmark results, broken scripts
- **Documentation improvements**

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full guidelines.

Quick summary:
1. Fork this repository
2. Create a branch: `git checkout -b algo/your-algorithm-name`
3. Add your algorithm as `algorithms/M{NN}_family_descriptor.py`
4. Run `python benchmarks/run_benchmark.py` and include results in your PR
5. Open a pull request using the provided template

---

## Citation

If you use this dataset or baseline in your research, please cite:

```bibtex
@dataset{sawitmvc2026,
  title   = {SawitMVC-YOLO: Multi-View Oil Palm Bunch Counting Dataset},
  author  = {ULM-DS-Lab},
  year    = {2026},
  url     = {https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO},
  license = {CC BY-NC 4.0}
}
```

---

## License

This project is licensed under [CC BY-NC 4.0](LICENSE).
You may use, share, and adapt this work for **non-commercial purposes** with attribution.
