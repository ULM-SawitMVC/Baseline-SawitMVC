# Multi-View Oil Palm Fresh Fruit Bunch Counting with Detector-Aware Tree-Level Regression

## Abstract

Accurate fresh fruit bunch (FFB) counting is important for oil palm harvest planning, but field images are affected by occlusion, viewpoint variation, and repeated visibility of the same physical bunch across multiple views. This paper studies multi-view tree-level FFB counting using 953 trees, 3,992 side-view images, and 9,823 uniquely identified bunches collected from DAMIMAS and LONSUM plantations in Tanah Laut, South Kalimantan, Indonesia. Each tree is evaluated as a four-dimensional count vector for maturity classes B1, B2, B3, and B4. The practical baseline uses YOLOv26-medium detections, aggregates all detections from the same tree into tree-level features, and predicts class counts using Ridge regression. On the 141-tree test split, the best end-to-end configuration reaches 77.48% Class ±1 Acc, 32.62% Tree ±1 Acc, and 1.036 Macro MAE. In contrast, when ground-truth detections are used, an ElasticNet counter reaches 98.05% Class ±1 Acc and 92.20% Tree ±1 Acc. The 20.57 percentage-point gap in Class ±1 Acc indicates that the main remaining limitation is detector quality rather than the tree-level counting formulation itself.

## Index Terms

Oil palm, fresh fruit bunch, multi-view counting, object detection, YOLO, tree-level regression, maturity grading, agricultural computer vision.

## I. Introduction

Oil palm harvest planning depends on estimating both the number and maturity state of fresh fruit bunches on each tree. Manual inspection is labor-intensive, while image-based automation remains difficult because bunches can be occluded by fronds, partially visible, visually ambiguous, or observed from multiple viewpoints. Prior work on oil palm FFB detection and ripeness classification has shown that computer vision and deep object detectors are useful for this domain, but field visibility, maturity ambiguity, and counting remain practical challenges [8], [9], [10]. A single image may miss important evidence, but a multi-view acquisition protocol introduces another challenge: the same physical bunch may appear in more than one image.

Object detection is widely used in agricultural computer vision for fruit, flower, and crop monitoring tasks. Koirala et al. reviewed deep learning for fruit detection and yield estimation, emphasizing public datasets, standard metrics, and practical issues such as occlusion in tree-crop imaging [1]. Kamilaris and Prenafeta-Boldu also surveyed the broader use of deep learning in agricultural and food-production problems [2]. DeepFruits demonstrated the use of deep neural networks for fruit detection under field conditions [7]. YOLO-style detectors are relevant because they formulate detection as a fast single-stage prediction problem [3], and later YOLO variants were designed to improve the speed-accuracy trade-off for real-time detection [4].

Counting from multi-view images differs from standard single-image detection. A detector estimates visible objects per image, while the operational target is often the number of unique physical objects per plant, plot, or tree. This creates a duplicate-observation problem: multi-view coverage improves recall, but repeated visibility can lead to overcounting. Song et al. studied fruit recognition and counting from multiple images with the goal of preventing double counting across views while handling missed fruit due to occlusion [5]. Stein et al. used multiple-view geometry for mango detection, localization, and yield estimation, showing that multi-view tracking can address occlusion and support tree-level fruit counting [6]. These studies motivate the explicit separation between image-level detection and tree-level unique-count estimation.

This paper studies tree-level FFB counting from multi-view images. The objective is not only to detect visible bunches in individual images, but to estimate the unique number of B1, B2, B3, and B4 bunches per tree. This distinction is important because direct summation of image-level detections counts appearances rather than physical bunches.

The main contributions are as follows:

1. A multi-view oil palm FFB counting benchmark with 953 trees, 3,992 images, and 9,823 unique bunches.
2. A tree-level evaluation protocol with a 716/96/141 train/validation/test split.
3. A detector-plus-counter baseline using YOLOv26-medium detections and Ridge regression over tree-level features.
4. A controlled comparison of naive appearance counting, perfect-detection counting, and practical end-to-end YOLO-based counting.
5. Reproducible result files and scripts for comparing future methods under the same split and metrics.

## II. Methodology

### A. Dataset and Task Definition

The dataset contains 953 oil palm trees and 3,992 side-view images. The images were collected from DAMIMAS and LONSUM plantations in Kabupaten Tanah Laut, South Kalimantan, Indonesia. Each tree has four or eight side-view images. Each annotated physical bunch has a unique identity, enabling evaluation of both visible appearances and unique tree-level counts.

The maturity classes are B1, B2, B3, and B4. These classes represent maturity-related visual states. B1 is generally red, large, and lower on the tree; B2 is a black-red transition class; B3 is generally black and visually frequent; and B4 is smaller and often higher on the tree.

**Table I. Tree-level split and unique-bunch totals.**

| Split | Trees | Share | B1 | B2 | B3 | B4 | Total Bunches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 716 | 75.13% | 729 | 1,330 | 3,771 | 1,533 | 7,363 |
| Validation | 96 | 10.07% | 91 | 193 | 500 | 195 | 979 |
| Test | 141 | 14.80% | 117 | 257 | 742 | 281 | 1,397 |

**Fig. 1.** Unique FFB class distribution by tree-level split.  
Recommended file: `figures/paper/fig01_split_class_distribution.png`.

Let tree \(i\) contain a set of side-view images \(V_i\). For each maturity class \(c \in \{B1,B2,B3,B4\}\), the ground-truth count is \(y_{i,c}\), and the predicted count is \(\hat{y}_{i,c}\). The prediction target for one tree is therefore

\[
\mathbf{y}_i = [y_{i,B1}, y_{i,B2}, y_{i,B3}, y_{i,B4}],
\]

with corresponding prediction

\[
\hat{\mathbf{y}}_i = [\hat{y}_{i,B1}, \hat{y}_{i,B2}, \hat{y}_{i,B3}, \hat{y}_{i,B4}].
\]

The evaluation unit is the tree, not the individual image. This prevents a method from appearing strong by detecting many visible appearances while still producing an incorrect final tree-level count.

Multi-view acquisition improves coverage, but it also creates repeated observations. If \(a_{i,c}\) denotes the number of visible appearances of class \(c\) across all views of tree \(i\), naive appearance counting can be written as

\[
\hat{y}_{i,c}^{naive} = a_{i,c}.
\]

This estimate is biased upward when a physical bunch appears in several views. In the current dataset, naive summation overcounts the unique bunch total by approximately 1.83x. This motivates a tree-level aggregation model that learns to infer unique counts from multi-view evidence.

### B. Detection and Tree-Level Regression

The practical baseline uses YOLOv26-medium. The detector is trained on the official training split. Its validation-set performance is shown in Table II. These are detection metrics on the validation split, not tree-level counting metrics on the test split.

**Table II. Validation detection performance of YOLOv26-medium.**

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|:---:|---:|---:|---:|---:|---:|
| all | 1887 | 0.504 | 0.570 | 0.521 | 0.243 |
| B1 | 201 | 0.606 | 0.801 | 0.746 | 0.379 |
| B2 | 388 | 0.478 | 0.433 | 0.425 | 0.213 |
| B3 | 959 | 0.505 | 0.656 | 0.550 | 0.243 |
| B4 | 339 | 0.427 | 0.389 | 0.363 | 0.137 |

The detector is strongest on B1 and weakest on B4. B4 recall is 38.9%, so many B4 bunches are missed before the counter stage. This limitation is important because the counter can only use evidence produced by the detector.

For each tree, all detections from all side views are aggregated into a fixed-length feature vector. The baseline feature set, denoted \(F0\), has 13 dimensions:

\[
F0 = [s_{B1:B4}, m_{B1:B4}, \mu_{B1:B4}, n_{views}],
\]

where \(s_c\) is the total number of detections for class \(c\) across all views, \(m_c\) is the maximum number of detections for class \(c\) in any single view, \(\mu_c\) is the mean number of detections per view, and \(n_{views}\) is the number of available side views.

The controlled study evaluates eight feature banks, not only \(F0\) and \(F_{all}\). These banks are built by adding three detector-derived feature groups to the 13-dimensional \(F0\) baseline. The confidence group has 20 dimensions, the side-distribution group has 20 dimensions, and the spatial-size group has eight dimensions. The full bank \(F_{all}\) has 67 dimensions because it combines \(F0\), all three optional groups, and six global composition features.

For each class \(c\), the confidence group contains \(\text{conf\_sum}_c\), \(\text{conf\_mean}_c\), \(\text{conf\_max}_c\), the number of detections with confidence at least 0.5, and the number of detections with confidence at least 0.6. The side-distribution group contains the standard deviation, minimum, coefficient of variation, number of detected sides, and consistency score of per-side detections. The spatial-size group contains the mean normalized vertical center and mean normalized bounding-box area. The global composition group contains the total naive count, four class fractions, and the B3 fraction among B2 and B3 detections.

**Table III. Feature banks used by the tree-level counter.**

| Feature Bank | Dim. | Added Groups | Description |
|---|---:|---|---|
| \(F0\) | 13 | None | Per-class detection sum, maximum per side, mean per side, and number of sides |
| \(F0+\text{conf}\) | 33 | Confidence | Adds confidence sums, means, maxima, and high-confidence detection counts |
| \(F0+\text{spatial}\) | 21 | Spatial-size | Adds mean vertical position and mean bounding-box area per class |
| \(F0+\text{distrib}\) | 33 | Side distribution | Adds how detections are distributed across side views |
| \(F0+\text{conf}+\text{spatial}\) | 41 | Confidence, spatial-size | Tests whether confidence and spatial cues are useful without side-distribution features |
| \(F0+\text{conf}+\text{distrib}\) | 53 | Confidence, side distribution | Tests confidence and side-distribution cues without spatial-size features |
| \(F0+\text{distrib}+\text{spatial}\) | 41 | Side distribution, spatial-size | Tests side-distribution and spatial cues without confidence features |
| \(F_{all}\) | 67 | All groups plus global composition | Full feature bank with all optional groups, total naive count, class fractions, and B3/B2-B3 mixture ratio |

The repeated dimensions are intentional. For example, \(F0+\text{conf}\) and \(F0+\text{distrib}\) both have 33 dimensions, but they represent different evidence. Likewise, the two 41-dimensional banks test different two-group combinations. This design makes the controlled matrix an ablation over feature information, not just a comparison between a small and a large vector.

The counter maps the tree-level feature vector \(\mathbf{x}_i\) to a four-class count prediction:

\[
\hat{\mathbf{y}}_i = f_{\theta}(\mathbf{x}_i).
\]

The evaluated counters are Linear Regression, Support Vector Machine (SVM), Random Forest (RF), Ridge, and ElasticNet. The primary practical baseline is Ridge regression with the 67-dimensional \(F_{all}\) feature set.

All main counting results are evaluated on the official 141-tree test split. Detector performance in Table II is reported on the validation split because it is a detector-selection metric. The tree-level end-to-end results use cached YOLO detections and test-set ground truth.

The experiments are grouped into three tracks: simple counting with perfect detections, machine-learning counting with perfect detections, and practical end-to-end counting with YOLO detections. The controlled YOLO-based matrix evaluates eight feature sets, five counter models, two training strategies, and one fixed test set. The main comparison uses the `train_only` strategy with 716 training trees. The secondary `train_val` strategy uses 812 trees by adding the validation split to the training data.

### C. Evaluation Metrics

Class-level ±1 correctness for tree \(i\) and class \(c\) is defined as

\[
I_{i,c}^{\pm 1} =
\begin{cases}
1, & |\hat{y}_{i,c} - y_{i,c}| \leq 1, \\
0, & \text{otherwise}.
\end{cases}
\]

Class ±1 Acc is the average of this indicator over all test trees and all four classes:

\[
\text{Class } \pm 1 \text{ Acc} =
\frac{1}{4N}\sum_{i=1}^{N}\sum_{c=1}^{4} I_{i,c}^{\pm 1}.
\]

Tree ±1 Acc is stricter. A tree is counted as correct only when all four class predictions are within one bunch:

\[
\text{Tree } \pm 1 \text{ Acc} =
\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\left(\sum_{c=1}^{4} I_{i,c}^{\pm 1} = 4\right).
\]

Macro MAE is the mean absolute error averaged over classes:

\[
\text{Macro MAE} =
\frac{1}{4N}\sum_{i=1}^{N}\sum_{c=1}^{4}
|\hat{y}_{i,c} - y_{i,c}|.
\]

The signed bias for class \(c\) is

\[
\text{Bias}_c = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_{i,c} - y_{i,c}).
\]

Positive bias indicates overcounting, while negative bias indicates undercounting.

## III. Results and Discussion

### A. Detector and Counting Results

Table IV shows that direct summation of visible appearances performs poorly even when detections are perfect. The naive method reaches only 50.00% Class ±1 Acc and 6.38% Tree ±1 Acc. Simple correction methods perform much better, confirming that duplicate observation is the central difficulty under perfect detections.

**Table IV. Simple counting checks using ground-truth detections on 141 test trees.**

| Method | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|---|---|---:|---:|---:|
| Add all appearances without correction | 141 test | 50.00% | 6.38% | 2.142 |
| Global divisor | 141 test | 95.39% | 85.11% | 0.376 |
| Visibility-adaptive divisor | 141 test | 95.92% | 87.23% | 0.340 |

Table V evaluates the same counter list used in the end-to-end comparison, but with perfect ground-truth-derived detections. ElasticNet achieves the best Class ±1 Acc at 98.05%, while SVM gives the lowest Macro MAE at 0.266.

**Table V. Machine-learning counting with perfect detections.**

| Method | Features | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|---|---|---|---:|---:|---:|
| Linear Regression | F0, 13 dim | 141 test | 97.52% | 90.07% | 0.277 |
| SVM | F0, 13 dim | 141 test | 97.87% | 91.49% | 0.266 |
| Random Forest | F0, 13 dim | 141 test | 95.92% | 84.40% | 0.365 |
| Ridge | F0, 13 dim | 141 test | 97.70% | 90.78% | 0.275 |
| ElasticNet | F0, 13 dim | 141 test | 98.05% | 92.20% | 0.277 |

The practical setting uses real YOLO detections rather than ground-truth detections. Table VI compares the five counters under \(F0\). ElasticNet is the best \(F0\) model by Class ±1 Acc.

**Table VI. End-to-end counting with YOLO detections using \(F0\).**

| Counter | Features | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|---|---|---|---:|---:|---:|
| Linear Regression | F0, 13 dim | 141 test | 75.71% | 30.50% | 1.048 |
| SVM | F0, 13 dim | 141 test | 74.82% | 29.08% | 1.043 |
| Random Forest | F0, 13 dim | 141 test | 73.23% | 26.95% | 1.110 |
| Ridge | F0, 13 dim | 141 test | 76.06% | 28.37% | 1.053 |
| ElasticNet | F0, 13 dim | 141 test | 76.42% | 29.79% | 1.043 |

Table VII reports the corresponding \(F_{all}\) comparison. Ridge is the best practical configuration, reaching 77.48% Class ±1 Acc, 32.62% Tree ±1 Acc, and 1.036 Macro MAE.

**Table VII. End-to-end counting with YOLO detections using \(F_{all}\).**

| Counter | Features | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|---|---|---|---:|---:|---:|
| Linear Regression | F_all, 67 dim | 141 test | 73.23% | 27.66% | 1.092 |
| SVM | F_all, 67 dim | 141 test | 73.94% | 24.82% | 1.057 |
| Random Forest | F_all, 67 dim | 141 test | 74.47% | 27.66% | 1.064 |
| Ridge | F_all, 67 dim | 141 test | 77.48% | 32.62% | 1.036 |
| ElasticNet | F_all, 67 dim | 141 test | 75.89% | 29.08% | 1.059 |

**Fig. 2.** Detection and counting performance gap between perfect detections and practical YOLO detections.
Recommended file: `figures/paper/fig02_detection_gap.png`.

**Fig. 3.** Controlled `train_only` matrix of Class ±1 Acc over feature sets and counters.  
Recommended file: `figures/paper/fig03_controlled_matrix.png`.

### B. Practical Interpretation

Table VIII holds the counter fixed and changes only the feature set from \(F0\) to \(F_{all}\). The richer feature bank improves Ridge and RF, but it reduces performance for Linear Regression, SVM, and ElasticNet. Therefore, the correct claim is not that more features always help; the supported claim is that Ridge is the best current match for the full feature bank.

**Table VIII. Same-model change from \(F0\) to \(F_{all}\).**

| Counter | F0 Class ±1 Acc | F_all Class ±1 Acc | Delta | F0 Tree ±1 Acc | F_all Tree ±1 Acc | Delta | F0 MAE | F_all MAE | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LR | 75.71% | 73.23% | -2.48 pp | 30.50% | 27.66% | -2.84 pp | 1.048 | 1.092 | +0.044 |
| SVM | 74.82% | 73.94% | -0.89 pp | 29.08% | 24.82% | -4.26 pp | 1.043 | 1.057 | +0.014 |
| RF | 73.23% | 74.47% | +1.24 pp | 26.95% | 27.66% | +0.71 pp | 1.110 | 1.064 | -0.046 |
| Ridge | 76.06% | 77.48% | +1.42 pp | 28.37% | 32.62% | +4.26 pp | 1.053 | 1.036 | -0.017 |
| ElasticNet | 76.42% | 75.89% | -0.53 pp | 29.79% | 29.08% | -0.71 pp | 1.043 | 1.059 | +0.016 |

Table IX summarizes the main test-set results. The key comparison is between practical YOLO-based counting and perfect-detection counting. Ridge with \(F_{all}\) reaches 77.48% Class ±1 Acc using real detections, while ElasticNet reaches 98.05% with perfect detections. The 20.57 percentage-point gap points primarily to detector limitations.

**Table IX. Consolidated test-set summary.**

| Setting | Method | Features | Set | Class ±1 Acc | Tree ±1 Acc | Macro MAE | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Naive count check | Add all appearances | GT annotations | 141 test | 50.00% | 6.38% | 2.142 | +0.936 | +1.702 | +4.709 | +1.220 |
| Simple correction check | Global divisor | GT detections | 141 test | 95.39% | 85.11% | 0.376 | +0.135 | +0.135 | +0.262 | -0.064 |
| Visibility-pattern correction check | Visibility-adaptive divisor | GT detections | 141 test | 95.92% | 87.23% | 0.340 | +0.106 | +0.149 | +0.099 | -0.099 |
| Controlled \(F0\) winner | ElasticNet | F0, 13 dim | 141 test | 76.42% | 29.79% | 1.043 | +0.007 | -0.057 | -0.135 | +0.000 |
| Best compact-spatial alternative | ElasticNet | F0+spatial, 21 dim | 141 test | 76.77% | 31.21% | 1.039 | +0.014 | -0.064 | -0.156 | +0.035 |
| Primary practical baseline | Ridge | F_all, 67 dim | 141 test | 77.48% | 32.62% | 1.036 | +0.014 | -0.078 | -0.177 | +0.071 |
| Perfect-detection counter | ElasticNet | F0 GT detections | 141 test | 98.05% | 92.20% | 0.277 | -0.050 | +0.043 | -0.064 | -0.028 |

The results support five observations. First, duplicate observation is a large and measurable issue in multi-view FFB counting. Naive appearance summation fails because it counts repeated visibility rather than unique physical bunches. Second, tree-level counting is highly effective when detections are accurate. With perfect detections, simple correction already exceeds 95% Class ±1 Acc, and the best learned counter reaches 98.05%. Third, the practical bottleneck is detector quality. The validation detection table shows weak B4 recall and moderate overall mAP. Because the counter depends on detector output, missed detections and class errors directly reduce the final tree-level count. Fourth, feature richness and model choice must be evaluated together. \(F_{all}\) is not universally better; it helps Ridge and RF but hurts LR, SVM, and ElasticNet under the same test protocol. Fifth, the current best practical recommendation is Ridge with \(F_{all}\). ElasticNet with \(F0\)+spatial is a strong compact alternative, but the highest test-set Class ±1 Acc is achieved by Ridge with the full 67-dimensional feature set.

The current evidence should be interpreted with two limitations. The dataset comes from two plantations in one regency, so additional validation is needed before claiming broad generalization across plantation conditions, cultivars, cameras, seasons, and acquisition protocols. The current detector is also weak for B4 and visually ambiguous B2/B3 cases. Future work should investigate stronger B3/B4 detection, explicit multi-view association, geometry-aware aggregation, uncertainty-aware counting, and additional data from diverse environments.

## IV. Conclusion

This paper presents a reproducible study of multi-view oil palm FFB counting at tree level. The practical baseline uses YOLOv26-medium detections, tree-level feature aggregation, and Ridge regression to predict B1-B4 counts. This configuration achieves 77.48% Class ±1 Acc on the 141-tree test split. With perfect detections, an ElasticNet counter reaches 98.05% Class ±1 Acc, showing that the aggregation problem is largely manageable when the detector evidence is accurate. The dominant path for future improvement is therefore better detection and stronger use of multi-view evidence, especially for B3 and B4.

## Acknowledgment

To be completed with institutional, funding, plantation-access, and data-collection acknowledgments before submission.

## Reproducibility Notes

The draft values are tied to committed repository artifacts:

- Split metadata: `ground_truth/split_manifest.csv`
- YOLO cached detections: `predictions/y26mv2_per_tree/`
- End-to-end metrics: `results/e2e_per_tree/`
- Perfect-detection counter metrics: `results/e2e_upper_bound/`
- Controlled matrix: `results/experiments/counting_controlled_results.csv`
- Release claim checker: `benchmarks/check_release_claims.py`
- Paper figure generator: `scripts/generate_paper_figures.py`

Before submission, run:

```powershell
python benchmarks/check_release_claims.py
python scripts/generate_paper_figures.py
```

## References

[1] A. Koirala, K. B. Walsh, Z. Wang, and C. McCarthy, "Deep learning - Method overview and review of use for fruit detection and yield estimation," *Computers and Electronics in Agriculture*, vol. 162, pp. 219-234, Jul. 2019, doi: 10.1016/j.compag.2019.04.017.

[2] A. Kamilaris and F. X. Prenafeta-Boldu, "Deep learning in agriculture: A survey," *Computers and Electronics in Agriculture*, vol. 147, pp. 70-90, Apr. 2018, doi: 10.1016/j.compag.2018.02.016.

[3] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, real-time object detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779-788, doi: 10.1109/CVPR.2016.91.

[4] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, "YOLOv4: Optimal speed and accuracy of object detection," arXiv:2004.10934, 2020.

[5] Y. Song, C. A. Glasbey, G. W. Horgan, G. Polder, and G. W. A. M. van der Heijden, "Automatic fruit recognition and counting from multiple images," *Biosystems Engineering*, vol. 118, pp. 203-215, Feb. 2014, doi: 10.1016/j.biosystemseng.2013.12.008.

[6] M. Stein, S. Bargoti, and J. Underwood, "Image based mango fruit detection, localisation and yield estimation using multiple view geometry," *Sensors*, vol. 16, no. 11, Art. no. 1915, Nov. 2016, doi: 10.3390/s16111915.

[7] I. Sa, Z. Ge, F. Dayoub, B. Upcroft, T. Perez, and C. McCool, "DeepFruits: A fruit detection system using deep neural networks," *Sensors*, vol. 16, no. 8, Art. no. 1222, Aug. 2016, doi: 10.3390/s16081222.

[8] N. A. Prasetyo, Pranowo, and A. J. Santoso, "Automatic detection and calculation of palm oil fresh fruit bunches using Faster R-CNN," *International Journal of Applied Science and Engineering*, vol. 17, no. 2, pp. 121-134, 2020, doi: 10.6703/IJASE.202005_17(2).121.

[9] J. W. Lai, H. R. Ramli, L. I. Ismail, and W. Z. W. Hasan, "Real-time detection of ripe oil palm fresh fruit bunch based on YOLOv4," *IEEE Access*, vol. 10, pp. 95763-95770, 2022, doi: 10.1109/ACCESS.2022.3204762.

[10] M. Y. M. A. Mansour, K. D. Dambul, and K. Y. Choo, "Object detection algorithms for ripeness classification of oil palm fresh fruit bunch," *International Journal of Technology*, vol. 13, no. 6, pp. 1326-1335, 2022, doi: 10.14716/ijtech.v13i6.5932.
