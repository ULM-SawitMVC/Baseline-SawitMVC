# SawitMVC Baseline: Multi-View Oil Palm Bunch Counting and Maturity Classification

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dataset: Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO)

A reproducible research baseline for counting and grading fresh fruit bunches on
oil palm trees from four to eight camera angles per tree. The repository bundles
every artifact needed to reproduce its published numbers from a fresh clone:
953 ground-truth annotations, three retrained YOLOv26 detectors, three saved
machine-learning counters, and 36 pre-computed result folders. Hugging Face
credentials are required only when retraining a detector.

---

## Abstract

Counting fruit bunches on an oil palm tree from a single photograph is
unreliable because each tree carries between zero and twenty mature bunches
distributed unevenly around its crown. The conventional remedy — photographing
the same tree from multiple angles and summing per-image detections — is worse
still: a single bunch routinely appears in two or more views, so the naive
total overshoots the true count by roughly eighty-three per cent on this
dataset. The SawitMVC baseline addresses the resulting deduplication problem
on **953 trees** (3,992 images, 9,823 unique bunches) from two commercial
plantations in Kabupaten Tanah Laut, Kalimantan, Indonesia.

The repository contributes three families of solutions and a clean upper-bound
study. *Track A* applies five deterministic heuristics directly to the ground
truth and reaches **87.62% within-one Acc** without any learning. *Track B*
trains a 13-dimensional feature pipeline plus a regression counter (SVM,
Random Forest, or Linear Regression) on noisy YOLOv26 detections; the best
configuration reaches **70.79% Acc±1**. *Track C* repeats Track B on features
derived from the perfect ground truth annotations and reaches **97.37% Acc±1**,
identifying the gap of **26.58 percentage points** as detector error, not
counter error. The detector is therefore the highest-leverage improvement
target. Every numerical claim in this document is backed by a clickable
artifact in [`results/`](results/), [`models/`](models/), or
[`ground_truth/`](ground_truth/).

---

## 1. Introduction

### 1.1 Motivation

Yield estimation for oil palm plantations in Kalimantan, Indonesia depends on
accurate per-tree counts across four maturity classes — ripe (B1), about to
ripen (B2), unripe (B3), and youngest (B4). Field inspectors today walk the
tree and tally by eye. Replacing the inspector with a camera and a model is
attractive but exposes a deduplication problem that has not been studied
in depth on a curated, multi-view dataset.

### 1.2 Problem statement

For each tree, the pipeline receives between four and eight images taken from
distinct compass angles. A perfect detector still sees the same physical bunch
from multiple angles, so naive aggregation across views overcounts. On the
953-tree SawitMVC release the naive sum baseline reaches only **3.78% Acc±1**
(see [`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv)),
while the best deduplication heuristic reaches **87.62%**.

### 1.3 Contributions

1. A reproducibility bundle: 953 ground-truth JSONs and the canonical split
   manifest, copied into the repository at
   [`ground_truth/`](ground_truth/) so that no external download is required
   to rerun any benchmark.
2. Five deterministic deduplication heuristics in [`algorithms/`](algorithms/),
   ranked by `Acc±1` in [`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv).
3. Three retrained YOLOv26 detectors at [`models/yolo/`](models/yolo/) (nano,
   small, medium) with their training logs at
   [`models/yolo/train_logs/`](models/yolo/train_logs/).
4. Three saved ML counter artifacts at [`models/counters/`](models/counters/) so
   end-to-end evaluation requires no retraining.
5. A 36-folder result archive at [`results/`](results/) covering every
   detector × counter combination across the per-tree, per-image, and
   ground-truth tracks.

---

## 2. Dataset

### 2.1 Source

The dataset is `ULM-DS-Lab/SawitMVC-YOLO`, released on Hugging Face. The
annotation half is mirrored in this repository under
[`ground_truth/`](ground_truth/). The image half (3,992 JPEGs, 960 × 1280) is
not bundled; it is required only when retraining a detector.

### 2.2 Statistics

| Statistic | Value | Evidence |
|-----------|------:|----------|
| Trees | 953 | [`ground_truth/split_manifest.csv`](ground_truth/split_manifest.csv) |
| DAMIMAS plantation | 854 | same |
| LONSUM plantation | 99 | same |
| Images | 3,992 | [`ground_truth/data.yaml`](ground_truth/data.yaml) |
| Unique bunches | 9,823 | derived from `summary.by_class` in each [`ground_truth/annotations/*.json`](ground_truth/annotations/) |
| Raw detections in GT | 18,544 | same |
| Deduplication ratio | 0.53 | naive sum overshoots by ~1.83× |
| Trees with 4 sides | 908 (95.3%) | derived from `images` keys |
| Trees with 8 sides | 45 (4.7%) | same |

### 2.3 Splits

The canonical split, stratified by `variety × dominant_class`, is fixed at
**763 train / 95 val / 95 test** trees. Every script reads
[`ground_truth/split_manifest.csv`](ground_truth/split_manifest.csv) and
honours the `new_split` column.

### 2.4 Maturity taxonomy

| Class | Visual cue | Position on tree | Role |
|:-----:|------------|------------------|------|
| B1 | Red, large, round | Lowest, ripest | Highest commercial value |
| B2 | Black with red transition | Above B1 | Imminent harvest target |
| B3 | Solid black, spiky, elongated | Above B2 | Schedule for next pass |
| B4 | Smallest, dark green-black | Highest, newest | Future inventory |

### 2.5 Annotation schema

Each per-tree JSON contains four nested structures: `images` (per-side
detections), `bunches` (human-confirmed unique bunches with `appearances`
across sides), `_confirmedLinks` (side-to-side bunch correspondences), and
`summary` (aggregated counts). The complete schema with field types and a
sample document is documented in
[`ground_truth/README.md`](ground_truth/README.md).

---

## 3. Method

### 3.1 Pipeline overview

The pipeline proceeds in three stages: detect bunches per image, count and
deduplicate across views, then evaluate the full end-to-end chain.

```
  Stage 1 — Detection
       images -->  YOLOv26 detector  -->  per-image detections
                                                 |
  Stage 2 — Counting                             v
                                       group by tree (4-8 sides)
                                                 |
                            +--------------------+--------------------+
                            |                                         |
                            v                                         v
              13-dim feature vector                       deterministic heuristic
                            |                                         |
                            v                                         v
                   ML counter (SVM/RF/LR)                  five M01..M05 rules
                            |                                         |
  Stage 3 — End-to-End      +--------------------+--------------------+
                                                 v
                                  per-class counts  [B1, B2, B3, B4]
```

**Stage 1 — Detection**

### 3.2 Detection — YOLOv26

Three model sizes were fine-tuned from COCO-pretrained YOLO26 weights on the
SawitMVC-YOLO release: nano [`models/yolo/y26n.pt`](models/yolo/y26n.pt), small
[`models/yolo/y26s.pt`](models/yolo/y26s.pt), and medium
[`models/yolo/y26m.pt`](models/yolo/y26m.pt). Each run used sixty epochs,
`batch=32`, `imgsz=640`, `patience=60`, `seed=42`, deterministic training, and
standard Ultralytics augmentation. The exact CLI invocation and per-epoch curves
are captured in [`models/yolo/train_logs/`](models/yolo/train_logs/).

**Stage 2 — Counting**

### 3.3 Track A — heuristic deduplication

Five deterministic algorithms in [`algorithms/`](algorithms/) consume the
detection list and produce per-class counts without any training. They share
three primitives — visibility count, adaptive correction, and max-per-side
selection — and differ only in how they combine them. The full design
rationale lives in [`docs/algorithms.md`](docs/algorithms.md); algorithm-level
notes are in [`algorithms/README.md`](algorithms/README.md).

### 3.4 Track B — per-tree ML counters

Detections for one tree are folded into a thirteen-dimensional feature vector
by [`pipeline/build_counting_features.py`](pipeline/build_counting_features.py):

```
[ naive_sum_B1..B4 ,  max_per_side_B1..B4 ,  mean_per_side_B1..B4 ,  n_sides ]
```

Three regressors are trained on this vector:

- SVM with an RBF kernel and `GridSearchCV` over `C` and `gamma` — saved at
  [`models/counters/svm.pkl`](models/counters/svm.pkl).
- Random Forest with `n_estimators=200`, `max_depth=10`, `random_state=42` —
  saved at [`models/counters/rf.pkl`](models/counters/rf.pkl).
- Linear Regression preceded by a `StandardScaler` — saved at
  [`models/counters/lr.pkl`](models/counters/lr.pkl).

All three artifacts were fitted on the y26s training split and can be reloaded
with `--load-model` to skip training entirely. Details and regeneration
commands are in [`models/counters/README.md`](models/counters/README.md).

### 3.5 Track C — ground-truth upper bound

The same counters are refitted on features derived from the ground-truth
annotations rather than from YOLO predictions. The resulting numbers in
[`results/e2e_upper_bound/`](results/e2e_upper_bound/) describe the ceiling
achievable if the detector were perfect, and serve as the reference for
isolating detector error from counter error.

**Stage 3 — End-to-End Pipeline**

### 3.6 Track B' — per-image variants

[`pipeline/run_e2e_per_image.py`](pipeline/run_e2e_per_image.py) runs the
detector image-by-image, then groups results into trees. Two counter families
apply:

- *Simple aggregation* — `max`, `mean`, `sum` over per-image counts. `max`
  approximates per-bunch deduplication; `sum` is the worst case (naive total).
- *ML or heuristic counter* — the same 13-dim features as Track B,
  reconstructed from the per-image grouping. Results are byte-equal to
  Track B because both pipelines derive identical features.

### 3.7 How the three stages connect

Stage 1 (Detection) establishes how well the YOLO model localises and
classifies bunches. Stage 2 (Counting) establishes how well algorithms
deduplicate across views given either perfect detections (Track A / Track C)
or noisy YOLO detections (Track B). Stage 3 (End-to-End) joins the two and
reports the combined system accuracy. Comparing Track B against Track C
isolates detector error from counter design error — a 26.58-point gap that
points to the detector as the highest-leverage improvement target.

---

## 4. Experimental setup

### 4.1 Hardware

Inference and counter evaluation require only a CPU. Retraining a YOLOv26
checkpoint needs a CUDA-capable GPU with at least 8 GB of memory. Counter
training takes seconds on CPU.

### 4.2 Training configuration

YOLOv26 — sixty epochs, `batch=32`, `imgsz=640`, `patience=60`, `seed=42`,
deterministic training, optimizer `auto`, COCO pretraining enabled, and standard
Ultralytics augmentation. The exact arguments appear in each
[`models/yolo/train_logs/y26{n,s,m}_train_log.txt`](models/yolo/train_logs/).

Counters — seed `42` in every script
([`pipeline/run_counting_svm.py`](pipeline/run_counting_svm.py),
[`pipeline/run_counting_rf.py`](pipeline/run_counting_rf.py),
[`pipeline/run_counting_lr.py`](pipeline/run_counting_lr.py)). The SVM grid
search uses `n_jobs=-1`; cross-validation scores can therefore drift by less
than half a per cent across machines but the test split metrics are stable.

### 4.3 Metrics

The primary metric is **Acc±1**: the fraction of trees where the predicted
count differs from the ground truth by at most one bunch, macro-averaged
across the four classes. The supporting metrics are macro MAE, total-count
MAE, and exact-profile accuracy (all four classes simultaneously correct).
Formal definitions and implementations are in
[`docs/evaluation.md`](docs/evaluation.md).

---

## 5. Results

**Stage 1 — Detection**

### 5.1 YOLOv26 detection — mAP50 on the val split

| Model | mAP50 | Inference speed | Size | Training log |
|-------|------:|----------------:|-----:|--------------|
| [`y26m.pt`](models/yolo/y26m.pt) | **0.528** | 1.0 ms | 42 MB | [log](models/yolo/train_logs/y26m_train_log.txt) |
| [`y26n.pt`](models/yolo/y26n.pt) | 0.515 | **0.3 ms** | **5.2 MB** | [log](models/yolo/train_logs/y26n_train_log.txt) |
| [`y26s.pt`](models/yolo/y26s.pt) | 0.511 | 0.4 ms | 20 MB | [log](models/yolo/train_logs/y26s_train_log.txt) |

`y26m` produces the best detector quality; `y26n` is the strongest size-speed
trade-off (recommended for deployment); `y26s` produces the strongest
end-to-end counter results despite a lower mAP50.

**Stage 2 — Counting**

### 5.2 Track A — heuristic deduplication on 953 trees

Evaluated on every tree in [`ground_truth/annotations/`](ground_truth/annotations/);
the full table including the naive sum baseline is in
[`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv).

| Rank | Algorithm | Acc±1 | Macro MAE | Approach |
|:----:|-----------|------:|----------:|----------|
| 1 | [`M01_selector_b2b3`](algorithms/M01_selector_b2b3.py) | **87.62%** | 0.3746 | Trifurcation selector plus B2 ↔ B3 correction |
| 2 | [`M02_selector_trifurc`](algorithms/M02_selector_trifurc.py) | 87.62% | 0.3757 | Trifurcation selector, base form |
| 3 | [`M03_blend_geometric`](algorithms/M03_blend_geometric.py) | 86.99% | 0.3767 | Geometric-mean blend |
| 4 | [`M04_blend_floor_clamped`](algorithms/M04_blend_floor_clamped.py) | 86.99% | 0.3848 | Floor-clamped weighted blend |
| 5 | [`M05_blend_vis_divide`](algorithms/M05_blend_vis_divide.py) | 86.99% | 0.3875 | Visibility plus adaptive divide |
| — | Naive sum baseline | 3.78% | 2.2867 | No deduplication |

### 5.3 Track C — ground-truth upper bound on 95 test trees

Counters trained on perfect ground-truth features; shows the ceiling achievable
if the detector were ideal.

| Counter | Acc±1 | Macro MAE | B1 | B2 | B3 | B4 |
|:-------:|------:|----------:|---:|---:|---:|---:|
| [LR](results/e2e_upper_bound/gt_lr/metrics.json) | **97.37%** | **0.276** | 100.0% | 97.9% | 93.7% | 97.9% |
| [SVM](results/e2e_upper_bound/gt_svm/metrics.json) | 96.58% | 0.300 | 100.0% | 96.8% | 92.6% | 96.8% |
| [RF](results/e2e_upper_bound/gt_rf/metrics.json) | 95.79% | 0.361 | 96.8% | 96.8% | 92.6% | 96.8% |
| [M01 on GT](results/heuristics_953/accuracy_full.csv) (953 trees) | 87.62% | 0.375 | — | — | — | — |

**Stage 3 — End-to-End**

### 5.4 Track B — per-tree end-to-end on 95 test trees

Twelve combinations (three detectors × four counters). Sorted by Acc±1.
Every row links to the underlying
[`results/e2e_per_tree/{detector}_{counter}/metrics.json`](results/e2e_per_tree/).

| Rank | Detector | Counter | Acc±1 | MAE | B1 | B2 | B3 | B4 |
|:----:|----------|---------|------:|----:|---:|---:|---:|---:|
| 1 | y26s | [SVM](results/e2e_per_tree/y26s_svm/metrics.json) | **70.79%** | 1.147 | 93.7% | 66.3% | 53.7% | 69.5% |
| 2 | y26m | [M01](results/e2e_per_tree/y26m_m01/metrics.json) | 69.21% | 1.295 | 91.6% | 69.5% | 48.4% | 67.4% |
| 3 | y26n | [SVM](results/e2e_per_tree/y26n_svm/metrics.json) | 68.95% | 1.168 | 91.6% | 68.4% | 54.7% | 61.1% |
| 4 | y26m | [SVM](results/e2e_per_tree/y26m_svm/metrics.json) | 68.95% | 1.168 | 93.7% | 70.5% | 50.5% | 61.1% |
| 5 | y26s | [LR](results/e2e_per_tree/y26s_lr/metrics.json) | 68.68% | 1.161 | 92.6% | 67.4% | 54.7% | 60.0% |
| 6 | y26n | [LR](results/e2e_per_tree/y26n_lr/metrics.json) | 68.16% | 1.171 | 92.6% | 72.6% | 53.7% | 53.7% |
| 7 | y26m | [LR](results/e2e_per_tree/y26m_lr/metrics.json) | 67.89% | 1.174 | 92.6% | 70.5% | 51.6% | 56.8% |
| 8 | y26m | [RF](results/e2e_per_tree/y26m_rf/metrics.json) | 66.84% | 1.216 | 90.5% | 64.2% | 54.7% | 57.9% |
| 9 | y26n | [RF](results/e2e_per_tree/y26n_rf/metrics.json) | 66.84% | 1.184 | 91.6% | 68.4% | 49.5% | 57.9% |
| 10 | y26s | [M01](results/e2e_per_tree/y26s_m01/metrics.json) | 64.21% | 1.313 | 89.5% | 55.8% | 49.5% | 62.1% |
| 11 | y26s | [RF](results/e2e_per_tree/y26s_rf/metrics.json) | 64.21% | 1.255 | 93.7% | 62.1% | 47.4% | 53.7% |
| 12 | y26n | [M01](results/e2e_per_tree/y26n_m01/metrics.json) | 63.95% | 1.342 | 89.5% | 64.2% | 43.2% | 58.9% |

### 5.5 Track B' — per-image end-to-end on 95 test trees

Twenty-one combinations (three detectors × seven counters). Two views of the
same numbers follow: a sorted simple-aggregation table, and a pointer to every
ML/heuristic folder.

Simple aggregation, sorted by detector:

| Detector | max | mean | sum |
|----------|----:|-----:|----:|
| y26m | [64.21%](results/e2e_per_image/y26m_max/metrics.json) | [55.79%](results/e2e_per_image/y26m_mean/metrics.json) | [50.00%](results/e2e_per_image/y26m_sum/metrics.json) |
| y26n | [63.68%](results/e2e_per_image/y26n_max/metrics.json) | [56.58%](results/e2e_per_image/y26n_mean/metrics.json) | [50.26%](results/e2e_per_image/y26n_sum/metrics.json) |
| y26s | [61.84%](results/e2e_per_image/y26s_max/metrics.json) | [53.95%](results/e2e_per_image/y26s_mean/metrics.json) | [52.63%](results/e2e_per_image/y26s_sum/metrics.json) |

`max` is the strongest simple aggregator: it counts a bunch once even if it
appears on several sides. `sum` is the worst: it counts the same bunch up to
eight times. ML/heuristic results per detector × counter combination are
stored at
[`results/e2e_per_image/{detector}_{counter}/metrics.json`](results/e2e_per_image/)
and are within numerical noise of their Track B counterparts because both
pipelines derive identical 13-dim feature vectors.

### 5.6 Cross-track summary

| Track | Best result | Acc±1 | Evidence |
|-------|-------------|------:|----------|
| A — heuristic on full GT | M01 / M02 | 87.62% | [`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv) |
| C — counter on GT features | LR | 97.37% | [`results/e2e_upper_bound/gt_lr/metrics.json`](results/e2e_upper_bound/gt_lr/metrics.json) |
| B — per-tree on YOLO | y26s + SVM | 70.79% | [`results/e2e_per_tree/y26s_svm/metrics.json`](results/e2e_per_tree/y26s_svm/metrics.json) |
| B' — per-image simple agg | y26m + max | 64.21% | [`results/e2e_per_image/y26m_max/metrics.json`](results/e2e_per_image/y26m_max/metrics.json) |

The Track B to Track C gap of **26.58 percentage points** is detector error,
not counter error.

---

## 6. Discussion

### 6.1 Detector bottleneck

The 26.58-point gap between Track B's best (70.79%) and Track C's best
(97.37%) is the most actionable finding in the repository. A perfect counter
operating on noisy detections cannot recover the truth that the detector has
already lost. Improving recall on partially occluded B3 bunches, in
particular, would close most of the gap.

### 6.2 Heuristic vs. learned counters

On noisy detections, the heuristic M01 and the learned SVM are within seven
percentage points of each other (64.21% vs. 70.79% on y26s); on ground-truth
features, the learned counter strongly dominates (97.37% vs. 87.62%). The
heuristic is conservative because it cannot exploit class-conditional
correlations; the learned counter exploits them when the detection signal is
clean enough to support them.

### 6.3 B2 ↔ B3 ambiguity

The transition from B2 (black with red residue) to B3 (uniform black) is
visually continuous. Even the heuristic-on-ground-truth track stops at
87.62%; the residual error is dominated by B2/B3 swaps that are not separable
without cross-view embeddings or higher-resolution colour information. This
ceiling is documented in [`docs/findings.md`](docs/findings.md).

### 6.4 Failure modes

The most informative failure modes are: under-counting B3 when bunches are
shaded by overlapping fronds, double-counting B1 when a single ripe bunch
splits into two detection boxes, and confusing B2 with B3 when colour
saturation is washed out. The per-tree predictions in every
[`results/*/predictions.csv`](results/) file are sufficient to reconstruct
each failure case.

---

## 7. Reproducibility

### 7.1 Prerequisites

- Python ≥ 3.10
- For retraining a detector: CUDA-capable GPU with ≥ 8 GB VRAM
- For everything else: CPU only

### 7.2 Installation

```
pip install -r requirements.txt
```

Dependencies are pinned to minimum versions in
[`requirements.txt`](requirements.txt).

### 7.3 One-command reproduction

```
bash scripts/reproduce_all.sh
```

The script chains Track A → Track B → Track B' → Track C → headline-claims
guard. It runs purely from the cached predictions in
[`predictions/`](predictions/) and the bundled GT in
[`ground_truth/`](ground_truth/); no Hugging Face access is required. Total
runtime on CPU is under thirty minutes.

### 7.4 Step-by-step

```
# Track A — five heuristics on the full 953-tree set
python benchmarks/run_benchmark.py

# Track B — per-tree end-to-end for one detector
python pipeline/run_e2e_pipeline.py --name y26s --skip-inference

# Track B' — per-image end-to-end for one detector
python pipeline/run_e2e_per_image.py --name y26s --skip-inference

# Track C — ground-truth upper bound
bash scripts/reproduce_upper_bound.sh
```

Individual counter scripts accept `--load-model models/counters/{svm,rf,lr}.pkl`
to reuse the bundled artifacts instead of refitting.

### 7.5 Verification

```
python benchmarks/check_release_claims.py
```

The script compares the heuristic top-5 numbers in
[`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv)
and the per-tree E2E winner in
[`results/e2e_per_tree/`](results/e2e_per_tree/) against the values quoted in
this README. It exits 0 on success and prints the offending rows on failure.

### 7.6 Full inference from images

Required only when a contributor wants to retrain or re-detect:

```
# 1. Download images from Hugging Face (annotations are already bundled)
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('ULM-DS-Lab/SawitMVC-YOLO', repo_type='dataset', \
    local_dir='./SawitMVC-YOLO', token=True)"

# 2. Per-tree inference for one detector
python pipeline/run_e2e_inference.py --name y26s --weights models/yolo/y26s.pt \
    --data ./SawitMVC-YOLO/
```

---

## 8. Repository layout

```
Baseline-SawitMVC/
├── README.md                         You are here
├── CONTRIBUTING.md                   Contribution rules and PR template
├── CHANGELOG.md                      Version history
├── LICENSE                           CC BY-NC 4.0
├── requirements.txt                  Python dependencies
│
├── ground_truth/                     Bundled annotations + split manifest
│   ├── README.md                     Schema, class taxonomy, provenance
│   ├── annotations/                  953 per-tree GT JSONs
│   ├── split_manifest.csv            train/val/test (763/95/95)
│   └── data.yaml                     YOLO class descriptor
│
├── algorithms/                       Track A — five deterministic heuristics
│   ├── README.md
│   └── M0[1-5]_*.py
│
├── pipeline/                         Track B / B' / C scripts
│   ├── README.md
│   ├── build_counting_features.py    13-dim feature library
│   ├── run_counting_{svm,rf,lr}.py   Counter trainers (support --save/--load-model)
│   ├── run_e2e_inference.py          YOLO per-tree inference
│   ├── run_e2e_pipeline.py           Per-tree end-to-end (Track B)
│   └── run_e2e_per_image.py          Per-image end-to-end (Track B')
│
├── benchmarks/                       Entry points
│   ├── README.md
│   ├── run_benchmark.py              Heuristic top-5 runner
│   └── check_release_claims.py       Headline-claims guard
│
├── results/                          Pre-computed evaluation outputs
│   ├── heuristics_953/               Track A (all 953 trees, 29 methods ranked)
│   ├── e2e_per_tree/                 Track B  (12 folders, 3 detectors × 4 counters)
│   ├── e2e_per_image/                Track B' (21 folders, 3 detectors × 7 counters)
│   └── e2e_upper_bound/              Track C  (3 folders, counters on GT features)
│
├── models/
│   ├── README.md
│   ├── yolo/                         Detector weights and training logs
│   │   ├── y26{n,s,m}.pt
│   │   └── train_logs/y26{n,s,m}_train_log.txt
│   └── counters/                     Saved ML counter artifacts
│       ├── README.md
│       └── {svm,rf,lr}.pkl
│
├── predictions/                      Cached YOLO outputs
│   ├── y26{n,s,m}_per_tree/          953 per-tree JSONs (one per detector)
│   └── y26{n,s,m}_per_image/         3,992 per-image JSONs (derived)
│
├── docs/                             Long-form documentation
│   ├── dataset.md
│   ├── algorithms.md
│   ├── training.md
│   ├── evaluation.md
│   ├── e2e_pipeline.md
│   └── findings.md
│
├── figures/                          Visualisations
│   ├── README.md
│   └── eda/
│
└── scripts/                          Orchestration shell scripts
    ├── reproduce_heuristics.sh
    ├── reproduce_e2e_per_tree.sh
    ├── reproduce_e2e_per_image.sh
    ├── reproduce_upper_bound.sh
    └── reproduce_all.sh
```

---

## 9. Contributing

The repository accepts three categories of contribution: a new deduplication
heuristic (must be deterministic, must beat M05's 86.99% Acc±1 on the full
953-tree set, must register itself in the [`algorithms/`](algorithms/)
package), a counter improvement on the existing 13-dim features, and bug
fixes or documentation. The detailed checklist, branch naming convention, and
PR template are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

A new algorithm typically requires four steps:

1. Implement `predict(detections) -> dict[str, int]` in
   `algorithms/M{NN}_{family}_{descriptor}.py`.
2. Register it in [`algorithms/__init__.py`](algorithms/__init__.py).
3. Run `python benchmarks/run_benchmark.py --save` and commit the updated
   [`results/heuristics_953/`](results/heuristics_953/) CSV.
4. Open a pull request that includes the new metrics row.

---

## 10. Citation

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

This project is licensed under [CC BY-NC 4.0](LICENSE). You may use, share,
and adapt the work for non-commercial purposes with attribution.
