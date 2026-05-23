# Benchmarking Multi-View Tree-Level Oil Palm Bunch Counting Under a Fixed Detector

## Abstract

Black Bunch Census/Black Bunch Count (BBC) supports oil palm harvest planning by estimating bunch counts by maturity stage. Image-based BBC is difficult because a single tree is observed from multiple side views: additional views reduce occlusion, but the same physical bunch may appear repeatedly and cause naive overcounting. This paper benchmarks multi-view tree-level fresh fruit bunch (FFB) counting for B1, B2, B3, and B4 counts under two detection conditions. In the GT detection setting, counters receive ground-truth boxes and classes, isolating the tree-level counting model. In the fixed-detector setting, counters receive YOLOv26-medium outputs, measuring the deployed detector-plus-counter pipeline. The benchmark uses 953 trees, 3,992 images, four maturity classes, 4-8 views per tree, and a fixed 716/96/141 train/validation/test split. With GT detections, ElasticNet reaches 98.05% Class +/-1 Acc and 92.20% Tree +/-1 Acc. Under the fixed detector, the best Ridge + F_all pipeline reaches 77.48% Class +/-1 Acc and 32.62% Tree +/-1 Acc. Feature ablation improves the best fixed-detector result only modestly over F0, so the main remaining error is detector output quality rather than the tree-level counting model.

## Index Terms

Oil palm, fresh fruit bunch, black bunch census, multi-view counting, object detection, YOLO, tree-level regression, agricultural computer vision.

## I. Introduction

Oil palm harvest planning depends on knowing not only how many fresh fruit bunches are present, but also how many belong to each operational maturity class. Field practice often uses Black Bunch Census/Black Bunch Count (BBC) to support production forecasting and harvesting decisions [11]. For this benchmark, the target is a four-dimensional tree-level count vector for B1, B2, B3, and B4.

Computer vision can reduce manual effort, but tree-level counting is not the same as image-level detection. A detector estimates visible bunch appearances in one image. BBC-style planning needs the number of unique physical bunches on a tree, separated by class. Multi-view imaging helps because a bunch hidden from one side may be visible from another, yet it also creates duplicate visibility. A naive sum over all side-view detections therefore counts appearances rather than bunch identities. Fig. 1 illustrates one B3 bunch visible in three adjacent views.

![Cross-view duplicate visibility of one B3 bunch](figures/paper/fig01_cross_view_linking.png)

**Fig. 1.** Cross-view duplicate visibility of one B3 bunch in sides 1-3 of `DAMIMAS_A21B_0847`.

Prior agricultural vision work has studied fruit detection and yield estimation with deep learning [1], [2], [7], including YOLO-style single-stage detectors [3], [4]. Multi-view fruit counting has also been studied as a way to reduce occlusion while avoiding double counting [5], [6]. Oil palm FFB detection and ripeness classification studies show the relevance of detectors in this crop, while also showing field challenges from visibility, maturity ambiguity, and class imbalance [8], [9], [10].

This paper frames multi-view oil palm FFB counting as a detector-vs-counter benchmark. The central question is whether errors mainly come from the tree-level counting model or from the detector evidence supplied to it. To answer that question, the same tree-level task is evaluated in two conditions:

1. GT detection setting: counters receive ground-truth detection evidence, isolating counting under ideal detector output.
2. Fixed-detector setting: counters receive cached YOLOv26-medium outputs, measuring the practical fixed-detector pipeline.
3. Feature ablation: the counter receives progressively richer tree-level features to test whether additional confidence, spatial, and side-distribution evidence closes the gap.

The result is a benchmark whose main takeaway is direct: the counting model is near ceiling when detections are accurate, while the fixed-detector pipeline drops sharply. Future progress should therefore prioritize detector quality and detector-aware multi-view evidence.

## II. Methodology

### A. Dataset and Task Definition

The benchmark uses 953 oil palm trees and 3,992 side-view images. Each tree has 4-8 side views. The four counted classes are B1, B2, B3, and B4. The fixed split contains 716 training trees, 96 validation trees, and 141 test trees. Dataset construction and annotation details are provided with the public dataset release.

For tree \(i\), the ground-truth target is

\[
\mathbf{y}_i = [y_{i,B1}, y_{i,B2}, y_{i,B3}, y_{i,B4}],
\]

and the predicted count vector is

\[
\hat{\mathbf{y}}_i = [\hat{y}_{i,B1}, \hat{y}_{i,B2}, \hat{y}_{i,B3}, \hat{y}_{i,B4}].
\]

The evaluation unit is the tree. This prevents image-level appearance counts from being mistaken for correct tree-level BBC estimates.

If \(a_{i,c}\) is the number of visible appearances of class \(c\) across all views of tree \(i\), naive multi-view counting is

\[
\hat{y}_{i,c}^{naive} = a_{i,c}.
\]

This is biased upward when one physical bunch is visible from multiple sides. In the test split, appearance summation reaches only 50.00% Class +/-1 Acc and 6.38% Tree +/-1 Acc even with GT annotations, which motivates learned tree-level aggregation.

### B. Detection Conditions

Fig. 2 summarizes the two detection conditions used in the benchmark. The GT detection setting supplies ground-truth boxes and classes to the counter. It measures how much error remains when detector output quality is removed as a source of failure. The fixed-detector setting supplies cached YOLOv26-medium outputs. It measures the full fixed detector-plus-counter pipeline while keeping the detector constant across counter models and feature banks.

![GT detection and fixed-detector pipeline diagram](figures/paper/fig02_detection_conditions.png)

**Fig. 2.** Two benchmark conditions: GT detection setting and fixed-detector setting.

The detector is strongest on B1 and weakest on B4. Table I reports validation detection performance for the fixed YOLOv26-medium detector.

**Table I. YOLOv26-medium validation detection performance.**

| Class | mAP50 | Recall |
|:---:|---:|---:|
| B1 | 0.746 | 0.801 |
| B2 | 0.425 | 0.433 |
| B3 | 0.550 | 0.656 |
| B4 | 0.363 | 0.389 |
| Overall | 0.521 | 0.570 |

### C. Counter Features and Models

For each tree, detections from all views are aggregated into fixed-length feature vectors. The baseline feature bank, F0, has 13 dimensions:

\[
F0 = [s_{B1:B4}, m_{B1:B4}, \mu_{B1:B4}, n_{views}],
\]

where \(s_c\) is the total count of detections for class \(c\), \(m_c\) is the maximum count in any side view, \(\mu_c\) is the mean count per side view, and \(n_{views}\) is the number of available views.

The controlled ablation evaluates F0, F0 + confidence, F0 + spatial, F0 + side-distribution, and combined feature banks up to F_all. Confidence features summarize detection scores; spatial features summarize vertical position and box area; side-distribution features summarize how detections vary across side views. F_all contains F0, all optional groups, and global composition features. The evaluated counters are Linear Regression, SVM, Random Forest, Ridge, and ElasticNet.

The counter maps a tree feature vector \(\mathbf{x}_i\) to the count vector:

\[
\hat{\mathbf{y}}_i = f_{\theta}(\mathbf{x}_i).
\]

### D. Evaluation Metrics

All counter models are trained on the 716-tree training split and evaluated on the fixed 141-tree test split.

Class-level +/-1 correctness for tree \(i\) and class \(c\) is

\[
I_{i,c}^{\pm 1} =
\begin{cases}
1, & |\hat{y}_{i,c} - y_{i,c}| \leq 1, \\
0, & \text{otherwise}.
\end{cases}
\]

Class +/-1 Acc averages this indicator over all test trees and classes:

\[
\text{Class } \pm 1 \text{ Acc} =
\frac{1}{4N}\sum_{i=1}^{N}\sum_{c=1}^{4} I_{i,c}^{\pm 1}.
\]

Tree +/-1 Acc requires all four class counts to be within one bunch:

\[
\text{Tree } \pm 1 \text{ Acc} =
\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}\left(\sum_{c=1}^{4} I_{i,c}^{\pm 1} = 4\right).
\]

Macro MAE is the mean absolute error over trees and classes. Signed class bias is the mean \(\hat{y}_{i,c} - y_{i,c}\), where positive values indicate overcounting.

## III. Results and Discussion

### A. GT Detection Setting

Table II merges simple duplicate-correction checks and learned counters under GT detections. Naive appearance summation fails because it counts repeated visibility. Simple correction already exceeds 95% Class +/-1 Acc, and learned counters reach 98.05% Class +/-1 Acc. This shows that once detector output quality is controlled, the tree-level counting problem is largely manageable.

**Table II. Counting with GT detections on the 141-tree test split.**

| Method | Type | Class +/-1 Acc | Tree +/-1 Acc | Macro MAE |
|---|---|---:|---:|---:|
| Naive sum | Heuristic | 50.00% | 6.38% | 2.142 |
| Global divisor | Heuristic | 95.39% | 85.11% | 0.376 |
| Visibility-adaptive divisor | Heuristic | 95.92% | 87.23% | 0.340 |
| ElasticNet | ML | 98.05% | 92.20% | 0.277 |
| SVM | ML | 97.87% | 91.49% | 0.266 |
| Ridge | ML | 97.70% | 90.78% | 0.275 |
| Linear Regression | ML | 97.52% | 90.07% | 0.277 |
| Random Forest | ML | 95.92% | 84.40% | 0.365 |

### B. Fixed-Detector Setting

Table III compares counters using only F0 under the fixed detector. The best F0 result is ElasticNet at 76.42% Class +/-1 Acc, while all five counters remain far below the GT detection setting.

**Table III. Fixed-detector counting with F0.**

| Counter | Class +/-1 Acc | Tree +/-1 Acc | Macro MAE |
|---|---:|---:|---:|
| ElasticNet | 76.42% | 29.79% | 1.043 |
| Ridge | 76.06% | 28.37% | 1.053 |
| Linear Regression | 75.71% | 30.50% | 1.048 |
| SVM | 74.82% | 29.08% | 1.043 |
| Random Forest | 73.23% | 26.95% | 1.110 |

### C. Feature Ablation

Table IV holds the counter family fixed to Ridge and changes only the feature bank. F_all is best, but the gain over F0 is only 1.42 percentage points in Class +/-1 Acc and 4.25 percentage points in Tree +/-1 Acc. Richer counter features help, but they do not close the GT-vs-detector gap.

**Table IV. Ridge feature ablation in the fixed-detector setting.**

| Features | Dim | Class +/-1 Acc | Tree +/-1 Acc | Macro MAE |
|---|---:|---:|---:|---:|
| F0 | 13 | 76.06% | 28.37% | 1.053 |
| F0 + confidence | 33 | 75.89% | 29.79% | 1.059 |
| F0 + spatial | 21 | 76.60% | 30.50% | 1.046 |
| F0 + side-distribution | 33 | 74.82% | 28.37% | 1.060 |
| F0 + confidence + spatial | 41 | 76.24% | 29.08% | 1.059 |
| F0 + confidence + side-distribution | 53 | 76.42% | 29.79% | 1.060 |
| F0 + side-distribution + spatial | 41 | 75.71% | 30.50% | 1.048 |
| F_all | 67 | 77.48% | 32.62% | 1.035 |

### D. GT vs Fixed-Detector Gap

Fig. 3 shows the per-class accuracy gap and fixed-detector bias. The largest accuracy loss is on B3, while the fixed detector also undercounts B2 and B3 on average. These errors are consistent with a detector-quality bottleneck: missed detections and class confusions reach the counter before any aggregation model can correct them.

![Per-class GT-vs-fixed-detector accuracy gap and fixed-detector bias](figures/paper/fig03_gap_bias.png)

**Fig. 3.** Per-class GT-vs-fixed-detector Class +/-1 Acc and fixed-detector bias.

Table V gives the central comparison. The GT detection setting is near ceiling with small signed biases. The fixed-detector pipeline has much lower Tree +/-1 Acc, despite using the strongest evaluated fixed-detector feature bank. The 20.57 percentage-point Class +/-1 Acc gap identifies detector output quality as the main remaining error source.

**Table V. Consolidated test-set summary.**

| Setting | Method | Class +/-1 Acc | Tree +/-1 Acc | Bias B1 | Bias B2 | Bias B3 | Bias B4 |
|---|---|---:|---:|---:|---:|---:|---:|
| GT detection | ElasticNet | 98.05% | 92.20% | -0.050 | +0.043 | -0.064 | -0.028 |
| Fixed-detector | Ridge + F_all | 77.48% | 32.62% | +0.014 | -0.078 | -0.177 | +0.071 |

The results support three conclusions. First, multi-view duplicate visibility is a real issue: appearance summation performs poorly even before detector errors are introduced. Second, tree-level counting is highly effective under GT detections, which means the counter formulation is not the dominant limitation. Third, fixed-detector output quality limits the deployed pipeline. Feature engineering and model choice provide modest gains, but the remaining gap is too large to attribute mainly to the regressor.

The main limitation of this benchmark is therefore the ceiling imposed by the fixed detector. YOLOv26-medium has weak B4 recall and moderate overall recall, and the counter cannot recover evidence that the detector never emits. Future work should improve B2/B3/B4 detection, calibrate detection confidence for counting, and use explicit cross-view association or geometry-aware aggregation where acquisition metadata supports it.

## IV. Conclusion

This paper benchmarks multi-view tree-level oil palm FFB counting under a fixed detector. The GT detection setting reaches 98.05% Class +/-1 Acc and 92.20% Tree +/-1 Acc with ElasticNet, showing that the tree-level counting model is near ceiling when detector evidence is accurate. The fixed-detector Ridge + F_all pipeline reaches 77.48% Class +/-1 Acc and 32.62% Tree +/-1 Acc, and feature ablation improves only modestly over F0. The main path to better BBC-style B1-B4 counting is therefore better detector output quality, followed by multi-view evidence use that is explicitly designed around detector uncertainty.

## Acknowledgment

To be completed with institutional, funding, plantation-access, and data-collection acknowledgments before submission.

## Code and Data Availability

Code: https://github.com/ULM-SawitMVC/Baseline-SawitMVC

Data: https://huggingface.co/datasets/ULM-DS-Lab/SawitMVC-YOLO

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

[11] M. Kerstan and T. Fairhurst, "Use of OMP for accurate crop forecasts," Agrisoft Systems and Tropical Crop Consultants Limited. Available: https://www.agrisoft-systems.com/wp-content/uploads/Papers/CropForecastsWithOMPBBC.pdf
