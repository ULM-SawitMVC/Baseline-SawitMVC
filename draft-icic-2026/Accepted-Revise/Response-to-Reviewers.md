# Response to Reviewers — ICIC 2026, Submission 57

**Title:** Benchmarking Multi-View Tree-Level Oil Palm Bunch Counting Under a Fixed Detector

We thank the reviewers for their constructive comments. We revised the manuscript
and audited the analysis code against the reported results. The main counting
scores reproduce exactly. The revision adds uncertainty estimates, complete paired
counter comparisons, per-class and total-count errors, appearance-level matching,
matched resampling, and sensitivity analyses. It also removes interpretations that
the experiments cannot establish.

Sections and tables refer to `main-new.tex`. The revised paper is eight pages.
The diagnostic figure is now Fig. 2; the redundant two-condition schematic was
replaced by a description in Sec. II-B. Tests and artifacts are supplied with the
revision in `experiments/revision/` and `results/revision/`. A separately trained
YOLO11m now supplies the requested second-family evaluation. Physical camera-pitch
changes remain untested; synthetic feature shifts are not presented as substitutes.

## Reviewer 1

> Expand the conclusion with implications, benefits, shortcomings, and future research.

**Addressed — Sec. IV and Sec. III-G.** The conclusion now explains how paired
GT-input and detector-input evaluations help practitioners separate counter
behaviour from input quality. It recommends investigating recall and maturity
confusion alongside aggregation, reports both class allocation and inventory error,
and describes the reproducibility benefits of cached predictions and fixed splits.
It states the limits of one dataset, two detector families, and two unequal estates.
Future work includes class-specific detector improvement, explicit association,
broader detector evaluation, independently acquired plantation data, calibrated pitch
tests, and repeated detector training. We do not claim a numerical ceiling on
improvements from future counters.

## Reviewer 2

> Single detector, limited test set, absence of significance analysis, and limited
> error decomposition weaken the broader conclusions.

**Addressed through additional analysis and narrower claims — Sec. II-E, III-A–G.**

- The reported-best GT–fixed Class ±1 gap remains **20.57 pp**, with tree-level
  95% bootstrap CI **17.20–23.94** and an exact paired permutation p < 0.001.
  Holding the model and features identical (Ridge+F0) gives **21.63 pp**,
  CI **18.44–25.00**, p < 0.001. Thus the gap does not require comparing different
  counter families.
- A new validation-selection check fits 40 model/feature candidates per input
  condition on 716 training trees and ranks them using only the 96 validation
  trees. The selected Ridge+F0 (GT) and SVM+F0+confidence (fixed) reach **97.70%**
  and **74.11%** Class ±1 on the 141 test trees: gap **23.58 pp**, paired 95% CI
  **20.21–27.13**, exact p < 0.001. The same historical test set is reused, so
  this addresses dependence on the selection rule, not independent replication.
- All ten counter pairs per condition are now tested on Class ±1, with Holm
  correction within each condition. None is significant under the fixed detector.
  The largest difference, ElasticNet versus RF, is +3.19 pp, raw p = 0.0628,
  adjusted p = 0.628. This is not evidence of model equivalence.
- Matched repeated CV uses identical trees, folds, training sizes, model and
  feature family under GT and fixed inputs. On 237 trees outside detector gradient
  training, Ridge+F0 reaches **97.72 ± 1.02%** and **72.70 ± 2.35%** respectively
  (mean ± fold SD, 25 folds). This pool includes detector validation trees; a
  stricter check using only the 141 original test trees reaches **97.16 ± 1.23%**
  versus **74.08 ± 2.96%**. The detector remains frozen. Smaller counter training
  sets and overlapping folds prevent treating these as independent full-pipeline
  validations or proving that the original test gap is conservative.
- Appearance matching separates unmatched annotations, incorrect classes among
  matched boxes, and unmatched detections. Diagnostic identity-based counting
  rules make their effects visible but do not form an additive decomposition of
  deployed error or an upper bound on counter improvement.
- Four YOLO26 checkpoints provide within-family sensitivity evidence. A new
  YOLO11m trained on the official split reaches **73.76%** Class ±1 with Ridge+F0,
  versus **76.06%** for YOLO26m and **97.70%** for GT using that same counter.
  The GT–YOLO11 gap is **23.94 pp** (95% CI **20.57–27.30**, p < 0.001).
  YOLO11–YOLO26 is −2.30 pp (CI −4.96 to +0.35, exact p = 0.118). The additional
  family supports the GT advantage without establishing model equivalence or a
  universal architecture ranking.

## Reviewer 3

### 1. Statistical uncertainty and significance

**Addressed — Sec. II-E, III-A–C; Tables III–V.** Percentile bootstrap intervals use
10,000 resamples of trees, seed 42, preserving the four classes within a tree.
Paired intervals resample the same indices for both configurations. Confidence
intervals for the main fitted configurations, including per-class metrics, are in
`a1_bootstrap_ci.json`; selected intervals are printed in the paper.

P-values in the revised discussion use an **exact paired permutation test** on the
per-tree number of correctly counted classes, rather than the previous bootstrap
tail fractions. `a10_revision_audit.py` computes the exact sample-exchange null
distribution by dynamic programming and tests all ten pairs per condition. Holm
adjustment is applied separately within each condition and to the seven Ridge
feature comparisons with F0. Exact McNemar tests concern Tree ±1, and Wilcoxon
tests concern per-tree macro absolute error; those endpoints are not conflated
with Class ±1. Raw and adjusted endpoint-specific p-values are available in
`a10_exact_paired_tests.csv`.

Under GT inputs, ElasticNet exceeds RF on Class ±1 (2.13 pp; p = 0.0042,
adjusted p = 0.0376), as does SVM (adjusted p = 0.0342). The four non-RF models
have no significant pairwise Class ±1 differences. Under fixed inputs, none of
the ten Class ±1 comparisons is significant. Intervals and tests are conditional
on fitted configurations; highlighting the highest test score is acknowledged as
exploratory and does not constitute independent model selection.

### 2. Scope of the detector conclusion

**Addressed as a scope clarification — Sec. II-B, III-E, III-G; Table VI.**
The new first block of Table VI compares YOLO11m, YOLO26m and GT on the official
141 test trees, with Ridge+F0 fitted on the same 716 training trees. YOLO11m uses
60 epochs, batch 32, image size 640, seed 42, and 12 data-loader workers. The best
checkpoint is selected on the 96 validation trees. YOLO11m used Ultralytics
8.4.140, while the reference detector training log records 8.4.49; architecture
defaults also differ. The comparison evaluates these trained pipelines rather
than isolating architecture as the sole cause of a difference.

Three earlier YOLO26 checkpoints (n, s, m) and released y26mv2 are compared with
Ridge+F0 fitted on the same 590 trees and evaluated on 64 trees outside gradient
training under both historical split protocols. Their Class ±1 scores range from
70.70% to 73.83%; the corresponding GT score is 97.27%. The 23.44-point difference
between GT and released y26mv2 remains much larger than the observed 3.13-point
checkpoint spread.

These 64 trees include detector validation trees. Different capacities also
coincide with historical training-split differences. We therefore describe this
as a sensitivity comparison, not a leak-free independent architecture ranking.
The eleven-threshold sweep refits the counter on official training trees at each
threshold. Ridge+F_all never exceeds 77.48% Class ±1 in that sweep, although
threshold 0.10 improves observed Tree ±1 to 38.30% from 32.62%. The test-set sweep
does not select a new threshold for the headline baseline.

### 3. Missed detections, false positives, and class confusion

**Addressed — Table I and Sec. III-D.** Greedy confidence-ordered, class-agnostic,
one-to-one matching uses IoU ≥ 0.5. Of **2,612** annotated test appearances:

- **964 (36.9%)** are unmatched;
- **396 (15.2%)** are matched but assigned an incorrect class;
- **1,252 (47.9%)** are matched with the correct class.

Of **2,354** detections, **706 (30.0%)** are unmatched. There are **317 of 1,397
unique bunches (22.7%)** without any matched detection across all views. B2 has
209 class-confused appearances, including 158 assigned B3; B4 has 262 unmatched
appearances out of 455. The full confusion matrix is supplied in
`a2_confusion_matrix.csv`.

Unmatched annotations may reflect absence, low IoU, or matching competition;
we no longer claim that this procedure isolates pure localisation failure.
Table I specifies GT-class versus predicted-class denominators and distinguishes
pooled test recall from macro validation recall. The diagnostic cache and the
filtered threshold cache differ by one detection at 0.25; Table VI now discloses
this and reports recall recomputed from the actual filtered cache at each
threshold, rather than reusing the original cache's recall. For example, macro
appearance recall at 0.25 is 0.441 for the filtered pass versus 0.442 for the
original pass. These values have been checked through a14 matching counts.

### 4. Per-class MAE/RMSE and total-count errors

**Addressed — Sec. II-D and Table V.** Per-class accuracy, MAE, RMSE, and signed
bias are reported for the best observed configuration in each condition. Macro
RMSE averages the four class RMSEs. Total-count MAE is **0.979** for GT and
**1.787** for fixed inputs; total ±1 is **78.01%** and **48.94%**. These differ
from Tree ±1, which requires all four class counts to be within one simultaneously.
The release also contains total RMSE and exact profile accuracy. We removed the
unsupported interpretation of per-tree MAE as block-level forecast error.

### 5. Repeated splits, CV, or independent plantation

**Addressed by matched counter resampling; independent plantation remains unavailable.**
Sec. II-E specifies joint estate/dominant-class stratification, five folds, five
repeats, training sizes, and the frozen detector. Sec. III-E reports the matched
237-tree and stricter 141-tree analyses described above. Fold membership and
per-tree predictions are saved in the a10 artifacts.

Counter transfer with Ridge+F_all fits 641 DAMIMAS training trees and scores 24
LONSUM evaluation trees at 71.88%, compared with 72.92% when fitted on 75 LONSUM
training trees. The reverse transfer scores 68.78% on 213 DAMIMAS evaluation
trees. Training size and estate effects are confounded, and the detector has seen
training trees from both estates. These are counter-transfer diagnostics, not
evaluation of the full pipeline on a plantation excluded from training.

### 6. Feature definitions

**Addressed — Table II.** The table defines the 13-dimensional F0 and the 20
confidence, 8 spatial, 20 distribution, and 6 composition dimensions. Definitions
include normalisation of box height/area, thresholds, population SD over all
available views including empty views, epsilon, and empty-class defaults. Empty
confidence mean/max and area equal zero; empty mean height equals 0.5. All 67
dimensions match the feature extractor.

The 1.42-point gain of Ridge+F_all over F0 has CI −0.53 to +3.37, exact p = 0.201,
and adjusted p = 1 across the seven feature comparisons. F_all does not lead the
repeated CV ablation. This does not prove that other representations cannot help.

### 7. Random Forest

**Addressed — Sec. III-F.** We examine restricted prediction range and shrinkage
using slopes, prediction variance, and count-load groups. GT RF slopes are
0.754–0.888 versus 0.865–0.951 for the linear models. For B1, RF predicts at most
three bunches, versus a training maximum of five and test maximum of six; its
MAE is 0.149 compared with Ridge's 0.050. RF trails ElasticNet by 3.07 points
in the highest-load bin. These are diagnostics consistent with sparse high-count
examples and averaging in leaves, not proof that RF is inherently unsuitable or
that tuning cannot help. Fixed-input differences remain unresolved after correction.

### 8. Moderate generalisations

**Addressed throughout.** Abstract and conclusion restrict the finding to the
tested SawitMVC configurations. We distinguish empirical GT references from
mathematical bounds, nonsignificance from equivalence, sensitivity studies from
independent validation, and observed test maxima from independently selected
models. We removed claims that richer features contain no useful information,
that recall alone ranks class-count errors, that label ambiguity has been
quantified, and that the fixed-split gap is necessarily conservative.

### 9. Detector versus multi-view association/aggregation

**Addressed with explicit limits — Sec. III-D and Fig. 2(a).** The diagnostic rules are:

| Rule | Class ±1 Acc |
|---|---:|
| GT boxes/classes with fitted ElasticNet+F0 | 98.05% |
| Matched physical bunches counted once using GT classes | 86.17% |
| Same identities, confidence-weighted detector class vote | 71.81% |
| Class vote plus every unmatched detection counted separately | 62.23% |
| Deployed Ridge+F_all | 77.48% |
| Naive sum of detector appearances | 50.71% |

The three identity-based rules diagnose particular error mechanisms. They are
different estimators from the fitted GT and deployed counters. Unmatched detections
are not associated across views, and nonlinear ±1 accuracy makes changes depend
on the order of interventions. The learned counter can compensate statistically
for missed counts and class bias, explaining why it exceeds two diagnostic rules.

We removed the previous claim that 8.69 points is the maximum improvement
available to association/counter methods. The 86.17% rule uses GT classes and
rejects unmatched detections, so its difference from the deployed score is not
an association-only bound. We also removed the additive interpretation of 28.19
points versus the GT counter's 1.95-point loss. Isolating association error itself
requires implementing and intervening on an explicit association module.

### 10. Proofreading, equations, tables, citations

**Addressed.** We rebuilt and checked the six tables, specified accuracy units
and metric averaging, and retained the corrected global divisor results
(95.57% / 86.52% / 0.356). We excluded M01 from the manuscript comparison because
overlap between its historical 228-tree development snapshot and the current
test split could not be established. Its code and historical results remain
available. The retained global divisor is now evaluated under both inputs, with
its scale calibrated separately on the 716 training trees: k = 1.891 (GT) and
1.793 (fixed). Fixed-input Class ±1 / Tree ±1 / macro MAE are **70.21% / 22.70% /
1.188**. All missing entries in Tables III and VI have been computed from the
corresponding caches, including checkpoint detection counts, GT appearance
counts, and per-threshold recall. GT recall equals one by construction.

The Tree ±1 indicator renders correctly; the exact whole-tree gap rounds to
59.57 pp. References retain first-citation order and the unsupported software
version “26.0.0” is removed; the actual training log records Ultralytics 8.4.49.
T1 font encoding restores the Times-family body font under Tectonic. The
diagnostic graphic no longer presents unrelated estimators as an additive ladder.
All six table captions are now short; detailed explanations use sentence-case
notes or adjacent prose. Tables use 8-point text and consistent numeric alignment.
Fig. 2 now spans both columns with side-by-side panels and a separate row for the
deployed counter. GPT Image Gen supplied a layout reference; the published vector
figure is rendered from saved numeric data to preserve exact bar lengths and
curve coordinates. The generation prompt and numeric plotting source are supplied.
The anonymous companion is generated from the corrected main source and removes
the entire funding paragraph, not merely its heading.

## Reviewer 4

> Fix column-alignment and typesetting problems in Tables III and V.

**Addressed.** Tables use consistent numeric alignment, short feature-group labels,
short captions, 8-point table text, and explicit units/averaging conventions. We inspected the compiled PDF, including
the combined counter table, ablation table, per-class table, and feature table.

> Briefly evaluate an additional detector architecture.

**Addressed — Sec. II-B, III-E; Table VI.** The author completed YOLO11m training
and evaluation on a separate GPU machine. The revision includes the selected
checkpoint, all 953 tree prediction files, 60-epoch training logs, arguments,
per-tree counting predictions, confusion matrix, and paired statistics.

With the same Ridge+F0 protocol, YOLO11m reaches **73.76% Class ±1 / 26.24%
Tree ±1 / 1.087 macro MAE**, compared with **76.06% / 28.37% / 1.053** for
YOLO26m and **97.70% / 90.78% / 0.275** for GT. The Class ±1 difference between
the two detectors is not significant (−2.30 pp, CI −4.96 to +0.35, p = 0.118).
The GT–YOLO11m difference remains large (23.94 pp, CI 20.57–27.30, p < 0.001).
All comparisons use 141 test trees and counters fitted on 716 training trees.

The validation-selected checkpoint's metrics match epoch 16 (mAP50 0.52450,
mAP50–95 0.25077); training ran the planned 60 epochs. The SHA-256 checksum
matches the hash embedded in every prediction cache. `a13_gpu_result_audit.py`
independently refits the counters on CPU and reproduces every committed metric,
paired test, per-tree prediction, and confusion count. It also verifies the
953-tree / 3,992-image coverage and split labels.

This fulfils the requested additional architecture evaluation for this dataset.
One training run per detector, software/default differences, and reuse of the
historical test set still limit wider generalisation. Calibrated camera-pitch
testing remains separate and outstanding.

> Clarify robustness of vertical features to real-world camera pitch.

**Addressed through sensitivity analysis and explicit limitations — Sec. III-E,
III-G and IV; physical pitch validation remains future work.** We transform
mean vertical features with a ∈ [0.90, 1.10] and b ∈ [−0.10, 0.10], clip to [0,1],
and preserve missing-class sentinels. Ridge+F_all loses at most **1.42 pp**;
Gaussian noise at σ = 0.05 costs **0.53 pp** in the seeded run. The previous
simulation incorrectly moved sentinel values for undetected classes; corrected
outputs are now supplied. This is a synthetic feature-shift test, not calibrated
camera rotation, which can also affect box area, occlusion, and detector output.
Physical pitch robustness remains unverified and is identified as a limitation.
The dataset does not contain calibrated pitch measurements or repeated captures
at known angles, so the revision restricts the spatial-feature findings to the
observed acquisition conditions. We explicitly propose repeated captures at
measured camera-pitch angles followed by end-to-end detection and counting
evaluation as future work. This clarifies the extent of the available robustness
evidence without asserting invariance to camera pitch.

## Reproducibility

| Artifact | Purpose |
|---|---|
| `a1_metrics_significance.py`, a1 results | Extended metrics and bootstrap intervals; original bootstrap-tail tests retained as historical artifacts |
| `a2_error_decomposition.py`, a2 results | Matching, confusion, diagnostic rules; corrected unique-bunch counts |
| `a3_resampling.py`, a3 results | Original CV and counter transfer; unmatched CV pool comparisons are superseded by a10 |
| `a4_detector_sensitivity.py`, a4 results | Within-family checkpoint and threshold sensitivity |
| `a5_spatial_robustness.py`, a5 results | Corrected feature perturbation preserving missing-class sentinels |
| `a6_rf_diagnosis.py`, a6 results | Prediction ranges and load-bin diagnostics |
| `a7_heuristics_ci.py`, a7 results | Historical heuristics and global divisor |
| `a8_ablation_cv.py`, a8 results | Eight Ridge feature sets under repeated CV |
| `a9_low_threshold_sweep.py`, a9 results | Low-threshold inference sweep with per-threshold counter refitting |
| `a10_revision_audit.py`, a10 results | All paired exact tests, Holm correction, reproduced headline metrics, matched CV and per-tree predictions |
| `a11_validation_selection.py`, a11 results | Validation-only ranking of 40 candidates per condition; selected test predictions, paired interval and exact test |
| `a12_second_detector.py`, a12 results and YOLO11 checkpoint/cache | Completed second-family GPU training and paired evaluation |
| `a13_gpu_result_audit.py`, `a13_gpu_audit.json` | Checkpoint/cache integrity, split coverage, CPU reproduction of a12 metrics and paired tests, training diagnostics |
| `a14_complete_table_metrics.py`, a14 results | Global divisor fitted separately under each input condition; checkpoint/GT appearance counts and recomputed threshold recall; completes Tables III/VI |
| `Reviewer-4-GPU.ipynb`, `reviewer-gpu-bundle.zip` | Historical portable package; actual completed GPU run used the committed a12 script |
| `test_revision_audit.py` | Six checks: exact inference against SciPy, Holm, comparison coverage, matched CV, validation ranking, selected metrics |
| `scripts/generate_revision_figure.py` | Fig. 2, derived from saved metrics |
| `figures/paper/imagegen-figure-prompt.txt`, `fig04_imagegen.png` | Image Gen prompt and layout reference; quantitative manuscript figure uses the vector plot |
| `scripts/build_icic_revision.py` | Synchronized main/anonymous source and PDF builds |

The response describes the supplied local revision package. Upload/publication of
the new artifacts is a separate author submission step.
