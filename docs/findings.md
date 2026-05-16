# Research Findings

Key findings from the SawitMVC multi-view oil palm bunch counting research
(completed 2026-05-16, 953 trees, Brand-New-Dataset-YOLO).

---

## Finding 1: The Bottleneck Is the Detector, Not the Counter

**Summary:** End-to-end accuracy is limited by YOLO detection quality, not the
counting algorithm. Improving the detector is the highest-leverage path forward.

**Evidence:**

| Setup | Acc ±1 |
|-------|:------:|
| Perfect detections (GT) + SVM counter | **96.1%** |
| Perfect detections (GT) + heuristic M01 | **87.6%** |
| YOLO26m detections + SVM counter (best E2E) | 71.6% |
| YOLO26n detections + M01 heuristic | ~68–72% |

The 24-pp gap between GT-SVM (96.1%) and E2E-SVM (71.6%) is entirely due to
YOLO detection errors (missed detections, wrong class labels, false positives)
corrupting the feature vectors used by the counter.

**All 15 E2E combinations** (5 detectors × 3 counting methods) land in a narrow
64–72% Acc±1 range — confirming that counting method choice barely matters once
detector quality is fixed.

---

## Finding 2: B2↔B3 Ambiguity Is Irreducible Without Cross-View Embeddings

**Summary:** 10–12% of trees have B2↔B3 bunches that are visually indistinguishable
from any single camera angle. This is a genuine dataset property, not label noise.

**Evidence:**
- Manual audit (audit-JSON-01): label noise confirmed at ≈ 0%
- Per-class MAE for B3 is 0.757 — nearly 5× higher than B1 (0.181)
- If B3 MAE were 0 (perfect), total Macro MAE would still be 0.209 — above the 0.2 target
- Estimated theoretical ceiling for no-training algorithms: **~89.6% Acc±1**
- 99–118 trees are structurally ambiguous and cannot be classified correctly without
  cross-view or temporal features

**Implication:** The 90% Acc±1 target cannot be achieved under the 100%-heuristic
constraint. Any algorithm reporting >90% on this dataset either uses training-derived
parameters or overfits to a subset.

---

## Finding 3: Per-Tree Routing Beats Global Rules

**Summary:** Algorithms that select different estimators per tree type (selectors)
consistently outperform algorithms that apply a single global formula (blends).

**Evidence:**

| Family | Best Acc±1 | Strategy |
|--------|:----------:|---------|
| Selector | **87.62%** (M01) | Route to best estimator per tree |
| Blend | 86.99% (M03) | Apply same formula to all trees |
| Weight | 86.88% (M06) | Apply same Gaussian weighting |

**Why:** Different tree types have fundamentally different detection profiles:
- Dense B3-dominated trees: median of 3 estimators is most robust
- B1-rich sparse trees: adaptive divisor is most accurate
- Mixed trees: geometric mean blend is the safest fallback

No single estimator is optimal across all profiles.

---

## Finding 4: COCO Pretraining Is Not Required for This Domain

**Summary:** Training YOLO26s from scratch (random initialization) achieves mAP50=0.511,
which **exceeds** the COCO-pretrained vanilla YOLO26s (mAP50=0.506).

**Explanation:** Oil palm bunch images are close-up agricultural photographs — very
different from the COCO distribution. COCO weights do not transfer meaningful low-level
features for this task. The scratch model converges at epoch 57 (vs epoch 21 for
pretrained), but reaches a slightly higher final mAP50.

**Practical implication:** When retraining on a new plantation dataset, starting from
scratch is a valid choice and may produce better domain-specific representations.

---

## Finding 5: Augmentation Is Essential

**Summary:** Disabling augmentation causes catastrophic overfitting and drops mAP50
by 0.056 (from 0.506 to 0.465). The model overfits by epoch 6.

**Evidence:**
- `y26s`: mAP50=0.465, early stop at epoch 6 (overfitting from epoch 1)
- `y26s_vanilla`: mAP50=0.506, stable convergence to epoch 21

With only ~3,200 training images, the model cannot learn generalizable features
without augmentation. Default Ultralytics augmentation (HSV, flip, translate, scale,
mosaic) is necessary and sufficient.

---

## Finding 6: Simple Methods Generalize; Complex Methods Overfit

**Summary:** Algorithms that perform best on the 228-tree development set show the
largest regression on the 953-tree canonical dataset.

| Algorithm | 228-tree Acc±1 | 953-tree Acc±1 | Drop |
|-----------|:--------------:|:--------------:|:----:|
| M12_selector_overrides | 97.37% | 85.94% | −11.43 pp |
| M17_selector_regime | 96.05% | 85.62% | −10.43 pp |
| **M05_blend_vis_divide** | ~86% | **86.99%** | **stable** |
| **M01_selector_b2b3** | n/a | **87.62%** | — |

Narrow override rules and regime-specific corrections that boost performance on small
development sets consistently fail to generalize. Simple geometric reasoning (blends,
visibility weights) is more robust.

---

## Future Work

### High priority (high expected impact)

1. **Better detection backbone** — The 24-pp gap between GT and E2E performance
   makes the detector the primary target. Potential approaches:
   - Multi-scale fusion (FPN) specialized for close-up agricultural imagery
   - Self-supervised pretraining on unlabeled oil palm images
   - Larger training datasets from additional plantation sites

2. **Cross-view feature fusion** — A model that sees all N sides simultaneously
   (instead of treating each side independently) could resolve B2↔B3 ambiguity
   using relative spatial information.

3. **Domain adaptation** — The current models are trained only on DAMIMAS and LONSUM.
   Adding data from other plantation varieties and geographic regions would improve
   robustness.

### Medium priority (incremental improvement)

4. **Uncertainty quantification** — Identifying the ~10–12% of trees that are
   structurally ambiguous (likely B2↔B3 confusion) and flagging them for human review
   could improve practical accuracy in deployment.

5. **Semi-supervised annotation** — The current JSON ground truth uses manual cross-view
   linking. A semi-supervised pipeline could scale annotation to larger datasets.

6. **Temporal consistency** — If the same tree is photographed at multiple harvest
   cycles, temporal priors on bunch counts could improve accuracy.

### Low priority (diminishing returns)

7. Ensemble of top-5 heuristics — early experiments show at most +0.3 pp improvement
8. Hyperparameter search for existing methods — already near the algorithmic ceiling
9. Class rebalancing — B3 dominates at 51.6%; oversampling B1/B4 destabilizes training
