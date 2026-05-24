# SawitMVC-YOLO: Multi-View Oil Palm Bunch Counting Baseline

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Dataset: Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO)

## Abstract

SawitMVC-YOLO is a multi-view dataset and baseline for counting oil palm fresh fruit bunches (FFB; Indonesian: TBS) at tree level. The dataset contains **953 oil palm trees**, **3,992 images**, and **9,823 unique bunches** collected from DAMIMAS and LONSUM plantations in Kabupaten Tanah Laut, South Kalimantan, Indonesia. Each tree is photographed from four or eight side views. This makes the task different from ordinary single-image detection, because one physical bunch can appear in several images of the same tree.

The main task in this repository is therefore not only detecting bunches in images, but also estimating the final number of B1, B2, B3, and B4 bunches for each tree. The practical baseline follows a simple two-stage pipeline. First, YOLO26m (`y26mv2`) detects bunches in every view. Second, all detections from the same tree are summarized into tree-level features and counted using Ridge regression. The best current end-to-end configuration is **Ridge regression with the 67-dimensional `F_all` feature set**, achieving **77.48% Class ±1 Acc**, **32.62% Tree ±1 Acc**, and **1.036 Macro MAE** on the 141-tree test split.

The most important finding is that the counting step can work very well when detections are correct. With ground-truth detections, the best machine-learning counter reaches **98.05% Class ±1 Acc** and **92.20% Tree ±1 Acc**. The large gap between 98.05% and 77.48% shows that the main remaining limitation is detector quality, especially for B3 and B4 bunches. This repository provides the dataset split, detector weights, cached predictions, evaluation files, and reproduction scripts so future methods can be compared against the same baseline.

**Keywords:** oil palm; fresh fruit bunch; multi-view counting; object detection; YOLO; duplicate observation; maturity grading; agricultural computer vision

## Main Findings at a Glance

- **The latest official split is 75/10/15 at tree level:** 716 training trees, 96 validation trees, and 141 test trees.
- **Naive multi-view counting is not valid:** adding all visible bunch appearances overcounts the true unique bunch total by about **1.83x**.
- **The main practical baseline is one detector-plus-counter pipeline:** YOLO26m (`y26mv2`) followed by Ridge regression on `F_all`.
- **The best end-to-end result is 77.48% Class ±1 Acc:** this means the predicted class count is within one bunch of the ground truth for most class-level tree counts.
- **Controlled comparison matters:** with the same `F0` features, ElasticNet is best at **76.42%**; with the same `F_all` features, Ridge is best at **77.48%**.
- **More features do not help every model:** `F_all` improves Ridge and RF, but reduces LR, SVM, and ElasticNet on the same test split.
- **Counting itself is not the main bottleneck:** when perfect detections are used, the counter reaches **98.05% Class ±1 Acc**.
- **The strongest next research direction is better detection:** B4 has low validation recall, and B3 remains the hardest class in the final tree-level count.

## 1. Introduction

Oil palm harvest planning depends on knowing how many bunches are present on a tree and how mature those bunches are. In practice, this information is still difficult to obtain automatically. A single image may miss bunches because of occlusion, camera angle, lighting, or overlapping fronds. Taking several images from different sides helps reveal more bunches, but it also creates a new problem: the same physical bunch may be seen more than once.

SawitMVC-YOLO focuses on this multi-view counting problem. The goal is to estimate the number of unique B1, B2, B3, and B4 bunches for each tree. The unit of evaluation is the tree, not the individual image. This matters because a method can detect many bunches in each image but still produce the wrong final count if it simply adds repeated observations from multiple views.

This repository contributes:

1. A multi-view oil palm bunch dataset with 953 trees, 3,992 images, and 9,823 unique bunches.
2. A latest official 75/10/15 split with 716 train, 96 validation, and 141 test trees.
3. A practical baseline that uses YOLO26m detections and a tree-level Ridge regression counter.
4. A clear comparison between naive counting, counting with perfect detections, and counting with real YOLO detections.
5. Reproduction scripts, cached predictions, released weights, and result files for transparent comparison.

## 2. Related Work

Agricultural computer vision studies commonly use object detection to locate fruits, bunches, flowers, or other crop targets in field imagery [1]. YOLO-style detectors are widely used because they are efficient and can be trained for field conditions where lighting, occlusion, object scale, and background clutter vary strongly [2]. Multi-view crop analysis is also related to plant phenotyping and yield estimation, where several images or viewpoints are combined to improve coverage [3].

This README uses placeholder citations only. The references at the end must be replaced with verified literature before any manuscript submission. This repository does not claim that the work has been accepted, indexed, or reviewed by Scopus or any other bibliographic database.

## 3. Dataset and Counting Problem

### 3.1 Dataset Composition and Latest Split

The dataset contains **953 trees**, **3,992 images**, and **9,823 unique bunches**. The images were collected from two plantations, DAMIMAS and LONSUM, in Kabupaten Tanah Laut, South Kalimantan. Each tree is represented by four or eight side-view images. Each physical bunch has a unique identity, so the dataset can tell whether a bunch appears in one view or several views.

The latest official protocol uses a **75/10/15 tree-level split**. Because the total number of trees is 953, the exact split becomes **716 training trees**, **96 validation trees**, and **141 test trees**. The corresponding percentages are **75.13% / 10.07% / 14.80%**. The canonical split field is `new_split` in [`ground_truth/split_manifest.csv`](ground_truth/split_manifest.csv).

**Table 1. Latest tree split and unique-bunch totals from `split_manifest.csv`.**

| Split | Trees | Share | B1 | B2 | B3 | B4 | Total Bunches |
|-------|------:|------:|---:|---:|---:|---:|--------------:|
| Train | 716 | 75.13% | 729 | 1,330 | 3,771 | 1,533 | 7,363 |
| Validation | 96 | 10.07% | 91 | 193 | 500 | 195 | 979 |
| Test | 141 | 14.80% | 117 | 257 | 742 | 281 | 1,397 |

**Figure 1. Latest class distribution by split under the 75/10/15 protocol.**
![Class distribution by split](figures/class_dist_by_split.png)

### 3.2 Bunch Maturity Classes

Each bunch is annotated into one of four maturity-related classes. The classes are not only different in color. They also tend to appear at different vertical positions on the tree, which later becomes useful for counting.

| Class | Visual Description | Typical Position | Operational Role |
|:-----:|--------------------|------------------|------------------|
| B1 | Red, large, round | Lowest | Highest commercial value |
| B2 | Black with red transition | Above B1 | Imminent harvest target |
| B3 | Solid black, spiky | Above B2 | Next harvest schedule |
| B4 | Smallest, dark green-black | Highest | Future inventory |

The heatmaps below show that the class positions are not random. B1 is usually lower, while B3 and B4 appear more often in upper regions. This is one reason why vertical position is included in the final feature set.

**Figure 2. Spatial heatmaps by maturity class.**
![B1 spatial heatmap](figures/eda/spatial_heatmap_B1.png)
![B2 spatial heatmap](figures/eda/spatial_heatmap_B2.png)
![B3 spatial heatmap](figures/eda/spatial_heatmap_B3.png)
![B4 spatial heatmap](figures/eda/spatial_heatmap_B4.png)

### 3.3 Why Multi-View Images Cause Double Counting

Multi-view images are useful because they reduce the chance that a bunch is hidden from the camera. However, the same benefit also creates repeated observations. A bunch that is visible from two sides may be detected twice. A bunch visible from three sides may be detected three times. If all detections are added directly, the result counts appearances, not unique physical bunches.

In SawitMVC-YOLO, naive summation overcounts the unique bunch total by approximately **83%**, or about **1.83x**. This is the central reason why the repository focuses on tree-level counting after detection. The detector answers the question, "what bunches are visible in each image?" The counter answers the more important operational question, "how many unique bunches are on this tree?"

**Figure 3. Latest appearance-count distributions for four-side and eight-side trees.**
![Appearance distribution for four-side trees](figures/appearance_dist_4side.png)
![Appearance distribution for eight-side trees](figures/appearance_dist_8side.png)

## 4. Baseline Method

### 4.1 YOLO Detector

The detector is **YOLO26m** with weights identified as `y26mv2`. It was trained on the official training split for 60 epochs using batch size 32, image size 640, patience 60, and seed 42. The released weights are stored at [`models/yolo/y26mv2.pt`](models/yolo/y26mv2.pt).

**Table 2. Validation detection performance of YOLO26m (`y26mv2`).**

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|:-----:|----------:|----------:|-------:|------:|---------:|
| all | 1887 | 0.504 | 0.570 | 0.521 | 0.243 |
| B1 | 201 | 0.606 | 0.801 | 0.746 | 0.379 |
| B2 | 388 | 0.478 | 0.433 | 0.425 | 0.213 |
| B3 | 959 | 0.505 | 0.656 | 0.550 | 0.243 |
| B4 | 339 | 0.427 | 0.389 | 0.363 | 0.137 |

The detector is strongest on B1 and weakest on B4. B4 recall is only **38.9%**, meaning many B4 bunches are missed. B3 has more examples than the other classes, but it is still difficult because many B3 bunches are visually ambiguous or partially occluded. These detector errors are important because the counter can only use the evidence provided by the detector.

### 4.2 Tree-Level Counting Pipeline

The baseline counts at tree level. The pipeline is:

1. Run YOLO on every image of a tree.
2. Collect all detections from all views of that tree.
3. Summarize the collected detections into numerical features.
4. Use a regression model to predict four numbers: B1 count, B2 count, B3 count, and B4 count.

The main baseline uses **Ridge regression** as the counter. Ridge is a linear regression model with regularization. In simple terms, it learns how to combine the tree-level features while avoiding overly unstable coefficients. This is useful because the dataset has 716 training trees, which is enough for a compact linear model but not necessarily enough for very flexible models with many parameters.

### 4.3 Feature Sets Used by the Counter

The simplest feature set is called `F0`. It contains 13 values. For each maturity class, it records three quantities: the total number of detections across all views, the maximum number of detections in one view, and the average number of detections per view. The final value is the number of available views for the tree.

```text
naive_sum_B1..B4 | max_per_side_B1..B4 | mean_per_side_B1..B4 | n_sides
```

The main feature set is called `F_all`. It contains 67 values. It keeps all `F0` values and adds information that helps the counter interpret the detections more carefully:

- detector confidence summaries, because high-confidence and low-confidence detections should not always be treated equally;
- distribution across views, because a class detected consistently across several views gives a different signal from a class detected only once;
- average vertical position, because B1, B2, B3, and B4 tend to appear at different heights on the tree;
- average bounding-box area, because bunch size is related to maturity and visibility;
- class proportions, because the relative mixture of B1-B4 can help stabilize the final count.

The important point is that `F_all` does not change the detector. It only gives the counter a richer summary of the detections already produced by YOLO.

### 4.4 Metrics and Evaluation Protocol

All main results use the official 716/96/141 tree split and are reported on the 141-tree test split. The main baseline is trained on the training trees and evaluated on the test trees. Cached YOLO detections are available in [`predictions/y26mv2_per_tree/`](predictions/y26mv2_per_tree/) so the reported results can be reproduced without rerunning detector inference.

The main metric is **Class ±1 Acc**. For each class, a prediction is counted as correct if it is within one bunch of the ground truth. For example, if the true B3 count is 6, then predictions of 5, 6, or 7 are treated as correct for B3. Class ±1 Acc averages this score over B1, B2, B3, and B4.

**Tree ±1 Acc** is stricter. A tree is counted as correct only if all four class predictions are within one bunch at the same time. This metric is lower because one wrong class is enough to make the whole tree incorrect.

**Macro MAE** is the average absolute counting error across the four classes. Lower Macro MAE means the predicted counts are numerically closer to the true counts.

## 5. Experimental Setup

The repository reports one main practical baseline and several supporting checks. The main practical baseline is:

```text
YOLO26m detections -> F_all tree-level features -> Ridge regression -> B1-B4 tree counts
```

The supporting checks are included to explain the result, not to present many competing baselines. The first check uses ground-truth detections to show how severe duplicate observation is. The second check also uses ground-truth detections to show how accurate counting can become when detector mistakes are removed. The controlled YOLO-based comparison then asks a fairer model-selection question: if the model is fixed, what happens when the feature set changes from `F0` to `F_all`?

The controlled matrix is generated by [`experiments/exp_counting_controlled.py`](experiments/exp_counting_controlled.py) and saved to [`results/experiments/counting_controlled_results.csv`](results/experiments/counting_controlled_results.csv). It tests:

- 8 feature sets: `F0`, `F0+conf`, `F0+spatial`, `F0+distrib`, `F0+conf+spatial`, `F0+conf+distrib`, `F0+distrib+spatial`, and `F_all`;
- 5 counter models: Linear Regression (`LR`), SVM, Random Forest (`RF`), Ridge, and ElasticNet;
- 2 training strategies: `train_only` using 716 training trees, and `train_val` using 812 training plus validation trees;
- one fixed test set: the official 141 test trees.

The main comparison uses `train_only`. The `train_val` results are shown later as a secondary check because using validation data for training is useful for final fitting, but it should not replace the cleaner 716-tree training comparison.

## 6. Results and Interpretation

### 6.1 Simple Counting Checks with Perfect Detections

This first result asks a basic question: if every bunch is already known correctly from the ground truth, what happens when multi-view observations are counted in different ways?

The naive method simply adds all visible appearances from all views. It performs poorly because it counts repeated appearances of the same bunch. This confirms that multi-view counting cannot be solved by image-level detection followed by direct summation.

The next method divides each class by one constant estimated from the training set. The four constants are B1 = 1.986, B2 = 1.786, B3 = 1.795, and B4 = 1.655. These constants are then applied unchanged to the test set. In the codebase this method is stored as `M15`, but conceptually it is just a simple class-wise correction for average duplicate visibility.

The best hand-designed correction in the current release is stored as `M01`. It still follows the same general idea, but it chooses the correction more flexibly based on the observed visibility pattern of the tree.

**Table 3. Simple counting checks on ground-truth detections for 141 test trees.**

| Method | Internal ID | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|--------|:-----------:|-----|--------------:|--------------:|----------:|
| Add all appearances without correction | naive | 141 test | 50.00% | 6.38% | 2.142 |
| Divide each class by one training-set constant | M15 | 141 test | 95.39% | 85.11% | 0.376 |
| Visibility-pattern correction | M01 | 141 test | 95.92% | 87.23% | 0.340 |

The message from this table is direct. Duplicate observation is a serious problem, but it can be corrected well when detections are correct. The naive method reaches only **50.00% Class ±1 Acc**, while the four-constant correction reaches **95.39% Class ±1 Acc**. The full ranking of 29 counting rules is available in [`results/heuristics_953/accuracy_full.csv`](results/heuristics_953/accuracy_full.csv).

### 6.2 Machine-Learning Counting with Perfect Detections

The second result asks whether a learned counter can perform even better when the detector makes no mistakes. This is not the final practical setting, because real deployment uses YOLO detections. It is still useful because it separates the counting problem from the detection problem.

**Table 4. Machine-learning counting result when detections are perfect.**

| Method | Features | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|--------|----------|-----|--------------:|--------------:|----------:|
| Linear Regression | F0, 13 dim | 141 test | 97.52% | 90.07% | 0.277 |
| SVM | F0, 13 dim | 141 test | 97.87% | 91.49% | 0.266 |
| Random Forest | F0, 13 dim | 141 test | 95.92% | 84.40% | 0.365 |
| Ridge | F0, 13 dim | 141 test | 97.70% | 90.78% | 0.275 |
| ElasticNet | F0, 13 dim | 141 test | 98.05% | 92.20% | 0.277 |

The best result in this setting is **98.05% Class ±1 Acc** with ElasticNet on ground-truth-derived `F0` features. This shows that the tree-level counting problem is almost solved when the input detections are accurate. Therefore, the much lower end-to-end result should be interpreted mainly as a detector limitation, not as evidence that tree-level count regression is fundamentally failing.

### 6.3 Controlled YOLO Counting: What Question Is Being Answered?

The main practical result uses realistic YOLO detections. In this setting, the detector can miss a bunch, assign the wrong class, or produce a detection with uncertain localization. The counter receives this imperfect evidence and must still estimate the final B1-B4 tree counts.

The important documentation point is that there are two different comparisons:

- A **model comparison under one feature set** asks: if all counters receive `F0`, which counter is best?
- A **configuration ranking** asks: across all feature sets and counters, which full configuration is best?

These are related, but they are not the same claim. The older `F0` table is a fair model comparison under the 13-dimensional baseline feature set. The best-configuration table is a practical ranking over feature set plus model together. To avoid mixing those two meanings, the controlled matrix below reports `F0`, `F_all`, the same-model delta from `F0` to `F_all`, and the overall ranking.

**Point to take away:** the best practical configuration is still **Ridge + `F_all`**, but the feature conclusion is more nuanced. More features help Ridge and RF, but they do not help every algorithm.

**Table 5. Main answers from the controlled YOLO counting matrix.**

| Question | Controlled Setting | Answer | Class ±1 Acc | Practical Reading |
|----------|--------------------|--------|--------------:|-------------------|
| If every model uses only `F0`, who is best? | 5 models, 13 dims, 716 train trees | ElasticNet | 76.42% | Compact baseline features are already strong. |
| If every model uses `F_all`, who is best? | 5 models, 67 dims, 716 train trees | Ridge | 77.48% | Regularized linear regression uses the richer feature bank best. |
| What is the best `train_only` configuration overall? | 8 feature sets x 5 models | Ridge + `F_all` | 77.48% | This is the headline practical baseline. |
| Does `F_all` always improve the same model? | Same model, `F0` vs `F_all` | No | Mixed | Extra detector-derived features can add signal or noise. |

### 6.4 If All Models Use `F0`, Which Counter Is Best?

`F0` is the cleanest baseline feature set. It contains only count summaries: total detections per class, maximum detections on one side, mean detections per side, and number of sides. It does not use confidence, bbox size, vertical position, or distribution features beyond those basic counts.

This table answers a narrow but important question: when the feature set is fixed to `F0`, which counter learns the best mapping from YOLO detection counts to final tree counts?

**Table 6. Controlled `F0` model comparison, trained on 716 trees and tested on 141 trees. Bias is mean signed error.**

| Counter | Class ±1 Acc | Tree ±1 Acc | Macro MAE | B1 ±1 Acc | B2 ±1 Acc | B3 ±1 Acc | B4 ±1 Acc | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ElasticNet | 76.42% | 29.79% | 1.043 | 96.45% | 80.14% | 56.03% | 73.05% | +0.007 | -0.057 | -0.135 | +0.000 |
| Ridge | 76.06% | 28.37% | 1.053 | 95.74% | 80.14% | 56.03% | 72.34% | +0.028 | -0.035 | -0.128 | -0.007 |
| LR | 75.71% | 30.50% | 1.048 | 96.45% | 79.43% | 55.32% | 71.63% | +0.014 | -0.064 | -0.142 | +0.057 |
| SVM | 74.82% | 29.08% | 1.043 | 95.74% | 76.60% | 55.32% | 71.63% | +0.014 | -0.582 | -0.284 | -0.312 |
| RF | 73.23% | 26.95% | 1.110 | 95.74% | 73.76% | 56.03% | 67.38% | +0.028 | -0.163 | -0.227 | +0.121 |

**Point to take away:** under `F0`, the best model is **ElasticNet**, not LR. LR remains useful as a simple stored reference at **75.71%**, but the controlled matrix shows that ElasticNet reaches **76.42%** with the same 13 input features.

The class-level pattern is also clear. B1 is easy for all models because the detector sees B1 more reliably and B1 counts are lower. B3 is the hardest class: every `F0` model is only around **55-56% B3 ±1 Acc**. This is why small improvements in B3 matter more than they look from the class-level score alone.

### 6.5 If All Models Use `F_all`, Which Counter Is Best?

`F_all` is the 67-dimensional feature set. It includes `F0` plus confidence statistics, per-side distribution statistics, vertical centroid, average bbox area, total detection count, class fractions, and a B3-vs-B2+B3 mixture feature. It gives the counter more context, but it also gives the counter more ways to overreact to noisy detector outputs.

This table fixes the feature set to `F_all` and compares the same five counter families.

**Table 7. Controlled `F_all` model comparison, trained on 716 trees and tested on 141 trees.**

| Counter | Class ±1 Acc | Tree ±1 Acc | Macro MAE | B1 ±1 Acc | B2 ±1 Acc | B3 ±1 Acc | B4 ±1 Acc | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ridge | 77.48% | 32.62% | 1.035 | 97.16% | 82.98% | 57.45% | 72.34% | +0.014 | -0.078 | -0.177 | +0.071 |
| ElasticNet | 75.89% | 29.08% | 1.059 | 97.16% | 80.85% | 53.90% | 71.63% | +0.000 | -0.085 | -0.106 | +0.071 |
| RF | 74.47% | 27.66% | 1.064 | 96.45% | 76.60% | 55.32% | 69.50% | -0.028 | -0.099 | -0.199 | +0.199 |
| SVM | 73.94% | 24.82% | 1.057 | 97.16% | 80.14% | 50.35% | 68.09% | -0.035 | -0.482 | -0.348 | -0.199 |
| LR | 73.23% | 27.66% | 1.092 | 95.74% | 80.85% | 50.35% | 65.96% | +0.007 | -0.163 | -0.163 | +0.149 |

**Point to take away:** under `F_all`, **Ridge is clearly best**. It reaches **77.48% Class ±1 Acc**, **32.62% Tree ±1 Acc**, and **1.035 Macro MAE**. It also gives the strongest B2 and B3 accuracy in this table.

The reason Ridge works well here is practical rather than mysterious. `F_all` contains correlated features: count totals, means, confidence sums, class fractions, and distribution summaries often describe overlapping evidence. Ordinary LR has no penalty to stabilize those coefficients, while Ridge adds regularization. That regularization lets Ridge use the extra signals without letting individual noisy features dominate the fit.

### 6.6 Does `F_all` Improve the Same Model?

The fairest way to discuss feature impact is to compare the same model under `F0` and `F_all`. This is different from comparing "best `F0` model" against "best `F_all` model", because that changes two things at once. The table below holds the model fixed and changes only the feature set.

**Table 8. Same-model change from `F0` to `F_all`. Positive Class ±1 Acc and Tree ±1 Acc deltas are better; negative MAE delta is better.**

| Counter | F0 Class ±1 Acc | F_all Class ±1 Acc | Delta Class ±1 Acc | F0 Tree ±1 Acc | F_all Tree ±1 Acc | Delta Tree ±1 Acc | F0 MAE | F_all MAE | Delta MAE |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LR | 75.71% | 73.23% | -2.48 pp | 30.50% | 27.66% | -2.84 pp | 1.048 | 1.092 | +0.044 |
| SVM | 74.82% | 73.94% | -0.89 pp | 29.08% | 24.82% | -4.26 pp | 1.043 | 1.057 | +0.014 |
| RF | 73.23% | 74.47% | +1.24 pp | 26.95% | 27.66% | +0.71 pp | 1.110 | 1.064 | -0.046 |
| Ridge | 76.06% | 77.48% | +1.42 pp | 28.37% | 32.62% | +4.26 pp | 1.053 | 1.035 | -0.018 |
| ElasticNet | 76.42% | 75.89% | -0.53 pp | 29.79% | 29.08% | -0.71 pp | 1.043 | 1.059 | +0.016 |

**Point to take away:** `F_all` is not universally better. It helps Ridge the most, helps RF modestly, and hurts LR, SVM, and ElasticNet. The correct claim is therefore not "more features always improve counting." The correct claim is: **the best current model-feature pairing is Ridge + `F_all`**.

This matters for future work. If a new paper or experiment uses a different counter model, it should not assume that the 67-dimensional feature bank is automatically better. Feature richness and model regularization have to be evaluated together.

### 6.7 Full `train_only` Controlled Matrix

The table below shows the full `train_only` Class ±1 Acc matrix: 8 feature sets x 5 models. This is the main controlled comparison because it trains only on the official 716 training trees and evaluates on the 141 test trees.

**Table 9. Full `train_only` controlled matrix, Class ±1 Acc.**

| Feature Set | Dims | LR | SVM | RF | Ridge | ElasticNet |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| F0 | 13 | 75.71% | 74.82% | 73.23% | 76.06% | 76.42% |
| F0+conf | 33 | 74.11% | 74.11% | 74.29% | 75.89% | 76.24% |
| F0+spatial | 21 | 76.06% | 74.82% | 73.58% | 76.60% | 76.77% |
| F0+distrib | 33 | 75.89% | 73.76% | 72.70% | 74.82% | 75.35% |
| F0+conf+spatial | 41 | 75.00% | 75.00% | 74.82% | 76.24% | 76.24% |
| F0+conf+distrib | 53 | 73.58% | 73.05% | 74.82% | 76.42% | 76.42% |
| F0+distrib+spatial | 41 | 75.18% | 74.82% | 73.58% | 75.71% | 75.18% |
| F_all | 67 | 73.23% | 73.94% | 74.47% | 77.48% | 75.89% |

**Point to take away:** the strongest single pattern is that **spatial features are useful**, but only when the model can absorb them cleanly. `F0+spatial` is the best feature set for LR and ElasticNet, and it is the second-best overall configuration after Ridge + `F_all`.

The confidence-only feature group does not produce a clean improvement. This is plausible because confidence is a detector-internal score, not a direct count of unique bunches. A low-confidence detection can still correspond to a real bunch, and a high-confidence detection can still be a duplicate view of the same physical bunch. Distribution features also do not help consistently by themselves; they are most useful when combined with a model that regularizes the full feature bank.

**Table 10. Top 20 `train_only` configurations from all feature sets and models.**

| Rank | Feature Set | Counter | Dims | Class ±1 Acc | Tree ±1 Acc | Macro MAE | B2 ±1 Acc | B3 ±1 Acc | Bias B2 | Bias B3 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | F_all | Ridge | 67 | 77.48% | 32.62% | 1.035 | 82.98% | 57.45% | -0.078 | -0.177 |
| 2 | F0+spatial | ElasticNet | 21 | 76.77% | 31.21% | 1.039 | 81.56% | 56.74% | -0.064 | -0.156 |
| 3 | F0+spatial | Ridge | 21 | 76.60% | 30.50% | 1.046 | 81.56% | 55.32% | -0.071 | -0.128 |
| 4 | F0 | ElasticNet | 13 | 76.42% | 29.79% | 1.043 | 80.14% | 56.03% | -0.057 | -0.135 |
| 5 | F0+conf+distrib | ElasticNet | 53 | 76.42% | 29.79% | 1.053 | 80.85% | 53.90% | -0.078 | -0.128 |
| 6 | F0+conf+distrib | Ridge | 53 | 76.42% | 29.79% | 1.060 | 80.14% | 54.61% | -0.092 | -0.113 |
| 7 | F0+conf | ElasticNet | 33 | 76.24% | 29.79% | 1.057 | 81.56% | 51.77% | -0.050 | -0.142 |
| 8 | F0+conf+spatial | ElasticNet | 41 | 76.24% | 29.79% | 1.060 | 80.85% | 52.48% | -0.035 | -0.156 |
| 9 | F0+conf+spatial | Ridge | 41 | 76.24% | 29.08% | 1.059 | 81.56% | 52.48% | -0.071 | -0.149 |
| 10 | F0+spatial | LR | 21 | 76.06% | 29.08% | 1.043 | 80.14% | 55.32% | -0.099 | -0.092 |
| 11 | F0 | Ridge | 13 | 76.06% | 28.37% | 1.053 | 80.14% | 56.03% | -0.035 | -0.128 |
| 12 | F0+conf | Ridge | 33 | 75.89% | 29.79% | 1.059 | 79.43% | 53.19% | -0.043 | -0.135 |
| 13 | F0+distrib | LR | 33 | 75.89% | 29.08% | 1.059 | 80.14% | 54.61% | -0.113 | -0.135 |
| 14 | F_all | ElasticNet | 67 | 75.89% | 29.08% | 1.059 | 80.85% | 53.90% | -0.085 | -0.106 |
| 15 | F0+distrib+spatial | Ridge | 41 | 75.71% | 30.50% | 1.048 | 80.85% | 54.61% | -0.113 | -0.149 |
| 16 | F0 | LR | 13 | 75.71% | 30.50% | 1.048 | 79.43% | 55.32% | -0.064 | -0.142 |
| 17 | F0+distrib | ElasticNet | 33 | 75.35% | 29.08% | 1.057 | 79.43% | 53.19% | -0.064 | -0.135 |
| 18 | F0+distrib+spatial | LR | 41 | 75.18% | 29.79% | 1.057 | 80.14% | 53.90% | -0.163 | -0.156 |
| 19 | F0+distrib+spatial | ElasticNet | 41 | 75.18% | 29.08% | 1.060 | 80.14% | 52.48% | -0.071 | -0.142 |
| 20 | F0+conf+spatial | SVM | 41 | 75.00% | 26.95% | 1.037 | 78.72% | 53.19% | -0.504 | -0.340 |

**Point to take away:** the Top 3 are all regularized linear models. The best unregularized LR row is rank 10, and the best SVM row is rank 20. This supports the practical choice of Ridge or ElasticNet over more sensitive models for the current YOLO output.

### 6.8 Secondary Check: Training on Train + Validation

The `train_val` strategy trains on 812 trees: the 716 training trees plus the 96 validation trees. It still evaluates on the same 141 test trees. This is useful as a secondary check because more training data can sometimes stabilize a model. It is not the cleanest model-selection comparison, because the validation split is no longer held out.

**Table 11. `train_val` controlled matrix, Class ±1 Acc.**

| Feature Set | Dims | LR | SVM | RF | Ridge | ElasticNet |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| F0 | 13 | 76.42% | 74.82% | 74.65% | 76.42% | 76.42% |
| F0+conf | 33 | 74.65% | 74.11% | 75.35% | 75.71% | 76.24% |
| F0+spatial | 21 | 75.89% | 74.47% | 73.76% | 76.24% | 76.60% |
| F0+distrib | 33 | 75.89% | 74.11% | 72.52% | 75.18% | 74.65% |
| F0+conf+spatial | 41 | 75.18% | 75.18% | 74.11% | 75.71% | 76.24% |
| F0+conf+distrib | 53 | 75.00% | 73.40% | 74.11% | 75.89% | 75.89% |
| F0+distrib+spatial | 41 | 75.18% | 74.11% | 74.11% | 75.18% | 75.89% |
| F_all | 67 | 73.76% | 73.76% | 75.18% | 76.42% | 76.06% |

**Point to take away:** adding validation trees does not beat the `train_only` headline result. The best `train_val` row is **ElasticNet + `F0+spatial` at 76.60%**, below **Ridge + `F_all` at 77.48%**. This means the primary result is not simply caused by using less training data in other models.

The `train_val` result also reinforces the same interpretation: compact and spatially enriched feature sets are stable, while `F_all` only works best with the right regularization and training setup.

### 6.9 Testing Whether More Complex Counters Help

The next question is whether the current result can be improved simply by using a more complex counter. The v4 experiment expands the feature set beyond 200 values and tests tuned boosted-tree models, larger feature sets, and model stacking. These models are more flexible than Ridge, but flexibility does not automatically help when the input detections are noisy.

**Table 12. Test of whether a more complex counter improves the current YOLO-based result.**

| Approach | Class ±1 Acc |
|----------|--------------:|
| Ridge + F_all 67 dim, main baseline | 77.48% |
| ElasticNet + F0+spatial, best v4 run | 76.77% |
| Ridge + full v4 feature set | 76.24% |
| Stacking, four models | 76.06% |
| XGB-Optuna | 74.11% |

**Point to take away:** none of the more complex counter variants exceeds the Ridge + `F_all` result. This does not mean that better counting models are impossible. It means that, for the current detector output and current training size, adding model complexity is not the most effective path.

For B2, the correlation between naive YOLO detection count and ground-truth count is only **r = 0.421**. In plain terms, the detector count for B2 is not strongly aligned with the true B2 count, so the counter starts from limited evidence. A larger counter cannot reliably recover information that the detector failed to provide.

### 6.10 Consolidated Result Summary

The table below gathers the main test-set results in one place. It should be read from top to bottom as a story: naive multi-view summation fails, simple correction works with perfect detections, learned counting works very well with perfect detections, and the practical YOLO-based baseline is lower because detector errors remain.

**Table 13. Consolidated test-set results. Bias is mean signed error; positive values indicate overcounting.**

| Setting | Method | Features | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|-------|--------|----------|-----|--------------:|--------------:|----------:|--------:|--------:|--------:|--------:|
| Naive count check | Add all appearances | GT annotations | 141 test | 50.00% | 6.38% | 2.142 | +0.936 | +1.702 | +4.709 | +1.220 |
| Simple correction check | Four training-set constants (M15) | GT detections | 141 test | 95.39% | 85.11% | 0.376 | +0.135 | +0.135 | +0.262 | -0.064 |
| Visibility-pattern correction check | Visibility-pattern correction (M01) | GT detections | 141 test | 95.92% | 87.23% | 0.340 | +0.106 | +0.149 | +0.099 | -0.099 |
| Stored end-to-end reference | Linear Regression | F0, 13 dim | 141 test | 75.71% | 30.50% | 1.048 | +0.014 | -0.064 | -0.142 | +0.057 |
| Controlled `F0` winner | ElasticNet | F0, 13 dim | 141 test | 76.42% | 29.79% | 1.043 | +0.007 | -0.057 | -0.135 | +0.000 |
| Best compact-spatial alternative | ElasticNet | F0+spatial, 21 dim | 141 test | 76.77% | 31.21% | 1.039 | +0.014 | -0.064 | -0.156 | +0.035 |
| Primary practical baseline | Ridge | F_all, 67 dim | 141 test | 77.48% | 32.62% | 1.036 | +0.014 | -0.078 | -0.177 | +0.071 |
| Perfect-detection counter | ElasticNet | F0 GT detections | 141 test | 98.05% | 92.20% | 0.277 | -0.050 | +0.043 | -0.064 | -0.028 |

The key comparison is the gap between **77.48%** and **98.05%** Class ±1 Acc. The gap is **20.57 percentage points**. Since the 98.05% result uses perfect detections, this gap points mainly to detector error. The most important future improvement is therefore better detection and better use of multi-view evidence, especially for B3 and B4.

## 7. Discussion

The results lead to five practical conclusions.

First, multi-view duplicate observation is real and large. The dataset was collected from multiple sides to reduce missed bunches, but repeated visibility means direct summation is not a valid final counting method. The naive method counts visible appearances, not unique bunches.

Second, duplicate observation can be handled when detections are reliable. The four-constant correction already reaches 95.39% Class ±1 Acc with ground-truth detections, and the best learned counter reaches 98.05%. This means the tree-level aggregation problem is manageable when the evidence is correct.

Third, the practical limitation is detector quality. The current YOLO detector is not equally strong across all maturity classes. B1 is detected most reliably, while B4 has the weakest recall and B3 remains difficult in the final count. This explains why the main baseline reaches 77.48% rather than approaching the perfect-detection result.

Fourth, the controlled matrix shows that feature engineering and model choice cannot be interpreted separately. `F_all` is the best feature set only with Ridge in this experiment. With LR, SVM, and ElasticNet, `F_all` reduces Class ±1 Acc compared with `F0`. This means the added confidence, distribution, and spatial features carry useful information, but they also carry detector noise. A stable model can use the signal; a less suitable model can fit the noise.

Fifth, the best practical recommendations are straightforward:

- Use **Ridge + `F_all`** as the main benchmark result.
- Use **ElasticNet + `F0+spatial`** as a compact alternative when a smaller, more interpretable feature set is preferred.
- Use **ElasticNet or Ridge on `F0`** when only count summaries are available.
- Do not assume boosted trees, stacking, or larger feature banks will improve performance unless detector quality also improves.

The algorithm-specific behavior is consistent with this reading. LR is strong with compact features but degrades with all 67 features. SVM undercounts B2 and B3 strongly in the YOLO setting, especially under richer features. RF benefits from `F_all` compared with `F0`, but not enough to beat regularized linear models. Ridge is the best match for the full feature bank because it controls coefficient instability. ElasticNet is strongest on compact and spatial feature sets, but its sparsity pressure appears less helpful when all noisy feature groups are included.

## 8. Limitations and Future Work

The dataset comes from two plantations in one regency, so additional validation is needed before making broad claims across different plantation conditions. The current detector is also still weak for B4 and visually ambiguous B2/B3 cases. Because the practical counter depends on detector output, missed detections and wrong classes directly reduce final counting accuracy.

The current baseline uses engineered tree-level features. It does not explicitly match the same physical bunch across views, reconstruct 3D geometry, or use temporal information. Future work should therefore prioritize improved B3/B4 detection, better detector calibration, explicit multi-view association, geometry-aware aggregation, and additional data from different locations, varieties, camera devices, and acquisition protocols.

The reference list in this README is intentionally incomplete. Verified peer-reviewed references should be added before using this document as a manuscript submission draft.

## 9. Conclusion

SawitMVC-YOLO provides a reproducible benchmark for oil palm bunch counting from multi-view tree images. The main baseline is intentionally simple: YOLO26m detects bunches in each image, tree-level features summarize the detections, and Ridge regression predicts the final B1-B4 counts. This baseline reaches **77.48% Class ±1 Acc** on the 141-tree test split.

The strongest finding is that counting can be much more accurate when detections are correct. With ground-truth detections, the counter reaches **98.05% Class ±1 Acc**. The difference between these results shows that the next major gain should come from better detection and stronger multi-view evidence handling, especially for B3 and B4.

## 10. Reproducibility

Create an isolated environment and install dependencies. The results above were regenerated locally from a repository venv using Python 3.14, but the code supports Python 3.10+.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install optional packages for the older v3/v4 extended experiments:

```powershell
python -m pip install xgboost lightgbm optuna
```

Download the dataset for fresh inference:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ULM-DS-Lab/SawitMVC-YOLO', repo_type='dataset',
                  local_dir='./SawitMVC-YOLO', token=True)
"
```

Prepare weights if needed:

```bash
cp models/yolo/y26me60p60b32s42v2.pt models/yolo/y26mv2.pt
```

Run end-to-end detector-plus-counter inference:

```bash
python pipeline/run_e2e_pipeline.py \
    --name y26mv2 --weights models/yolo/y26mv2.pt \
    --data SawitMVC-YOLO/ --counters svm lr rf m01
```

Report metrics:

```bash
python scripts/report_metrics.py y26mv2 test
```

Run simple duplicate-correction checks and the perfect-detection counter:

```bash
python benchmarks/run_benchmark.py
bash scripts/reproduce_upper_bound.sh
```

Run the controlled counting matrix used in this README:

```powershell
python experiments/exp_counting_controlled.py
```

The script writes [`results/experiments/counting_controlled_results.csv`](results/experiments/counting_controlled_results.csv) with 80 rows: 8 feature sets x 5 models x 2 training strategies.

Run the older counting comparison and deep-probe experiments:

```powershell
python experiments/exp_counting_v3.py
python experiments/exp_counting_v4.py
```

Verify release headline claims:

```powershell
python benchmarks/check_release_claims.py
```

## 11. Repository Layout

```text
algorithms/          # Hand-designed duplicate-correction algorithms, M01..M05
pipeline/            # End-to-end and perfect-detection feature extraction plus ML counter scripts
experiments/         # Controlled counting matrix and older v3/v4 probes
benchmarks/          # run_benchmark.py and check_release_claims.py
ground_truth/        # 953 GT JSONs and split_manifest.csv
predictions/         # Cached YOLO outputs, including y26mv2_per_tree/
results/             # Pre-computed evaluations and experiment CSVs
models/yolo/         # y26mv2.pt and training artifacts
models/counters/     # Stored SVM, RF, and LR counters
scripts/             # Reproduction and reporting helpers
figures/             # Dataset and analysis figures
archive/             # Older experiments, not part of the latest baseline
```

## 12. Data, Model, and Code Availability

The dataset is available from Hugging Face at [`ULM-DS-Lab/SawitMVC-YOLO`](https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO). Repository artifacts include cached predictions, trained detector weights, counter outputs, evaluation CSVs, and reproduction scripts. The published release numbers in this README are tied to the committed result files and can be checked with [`benchmarks/check_release_claims.py`](benchmarks/check_release_claims.py).

## 13. Ethics and License

This repository is released for non-commercial research use under [CC BY-NC 4.0](LICENSE). Users are responsible for respecting plantation access permissions, data-use restrictions, and any institutional requirements that apply to downstream deployment.

## 14. Citation

```bibtex
@dataset{sawitmvc2026,
  title   = {SawitMVC-YOLO: Multi-View Oil Palm Bunch Counting Dataset},
  author  = {ULM-DS-Lab},
  year    = {2026},
  url     = {https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO},
  license = {CC BY-NC 4.0}
}
```

## References

[1] Placeholder reference on agricultural computer vision and fruit yield estimation. To be replaced with a verified peer-reviewed source.

[2] Placeholder reference on object detection methods for field crop or tree-crop imagery. To be replaced with a verified peer-reviewed source.

[3] Placeholder reference on multi-view counting, plant phenotyping, or duplicate-observation handling. To be replaced with a verified peer-reviewed source.
