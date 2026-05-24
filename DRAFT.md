# Benchmarking Multi-View Tree-Level Oil Palm Bunch Counting Under a Fixed Detector

Muhammad Zainal Muttaqin¹, Fatma Indriani¹, Setyo Wahyu Saputro¹, Alia Rahmi¹, Dwi Kartini¹, Triando Hamonangan Saragih¹, Naufal Said³, and Hartoni²

¹ Department of Computer Science, Faculty of Mathematics and Natural Sciences, Lambung Mangkurat University, Banjarbaru, Indonesia  
² Department of Agro-Industrial Technology, Faculty of Agriculture, Lambung Mangkurat University, Banjarmasin, Indonesia  
³ Department of Agribusiness, Faculty of Agriculture, Lambung Mangkurat University, Banjarmasin, Indonesia  
Corresponding author: Fatma Indriani, f.indriani@ulm.ac.id

## Abstract

Tree-level yield forecasting often requires a count of objects in several readiness classes rather than a single total. Multi-view imaging helps recover objects hidden by occlusion, but it also creates duplicate observations of the same physical object across views. A deployed system must therefore solve two linked problems: detecting objects in individual images and aggregating those detections into one count vector per tree. This paper benchmarks that interaction on 953 oil palm trees and 3,992 side-view images under two controlled detection conditions. When the counting model receives annotated detections, it reaches 98.05% accuracy at the class level and 92.20% at the whole-tree level. When the same counting task uses outputs from a fixed object detector, those accuracies drop to 77.48% and 32.62%. The 20.57 percentage-point class-level gap remains after testing multiple counting models and feature sets, indicating that detector quality is the limiting factor in this fixed-detector pipeline rather than tree-level count aggregation.

## Index Terms

Oil palm, fresh fruit bunch, black bunch census, multi-view counting, object detection, YOLO, tree-level regression, agricultural computer vision.

## I. Introduction

Oil palm harvest planning is driven by Black Bunch Census (BBC), a field practice in which counters estimate, for every tree, how many bunches belong to each operational maturity class: typically B1 (ripe, harvest now), B2 (imminent), B3 (next cycle), and B4 (future inventory) [12]. The decision-relevant signal is therefore not the total number of bunches on a tree, but the four-dimensional count vector across maturity classes. Image-based BBC must reproduce this per-class disaggregation rather than collapse it into a single total.

Tree-level counting is geometrically harder than image-level detection [1]. Bunches surround the full circumference of an oil palm, so a single image misses bunches because of occlusion, viewing angle, or overlapping fronds. Multi-view acquisition reduces missed bunches by photographing a tree from four or eight side views, but the same physical bunch then appears in several of those images. Fig. 1 illustrates the effect: one B3 bunch is visible, and would be detected, in three adjacent side views of tree `DAMIMAS_A21B_0847`. A naive sum over per-image detections therefore counts appearances, not bunch identities, and is biased upward by roughly a factor of two on this dataset.

![Cross-view duplicate visibility of one B3 bunch](figures/paper/fig01_cross_view_linking.png)

**Fig. 1.** A single physical B3 bunch on tree `DAMIMAS_A21B_0847` is visible across sides 1, 2, and 3. If every per-image detection is counted, three appearances are added for one bunch. Tree-level counting must therefore aggregate evidence across views rather than sum it.

Prior work frames the problem from three directions. Oil palm FFB detection and ripeness classification have been studied with both two-stage and YOLO-style detectors [9], [10], [11], [13]; these works report usable mAP on FFB but also document recurring weaknesses on partially occluded or maturity-ambiguous bunches.

Multi-view fruit aggregation has been used in fruit recognition and yield estimation, including methods that explicitly fuse evidence across views to avoid double counting [2], [5], [6]. Recent plant-phenotyping work has also treated sparse or calibrated viewpoints as an association problem, either by solving multi-view triangulation without known correspondences [3] or by contrasting few-view counting with denser 3D reconstruction [4]. Recent neural-field approaches extend this idea by counting fruit from unstructured multi-view orchard photographs in a shared 3D representation [8]. Comparative studies on apple orchards have further refined these pipelines [6], [7]. These methods, however, are evaluated mainly on smaller tree-fruit crops with simpler geometry than oil palm, and they do not isolate detector quality from counter quality.

Regression-based counting and yield estimation in agricultural vision are well established [1], [6], [7], and YOLO-style single-stage detectors are now standard front-ends for such pipelines [14], [15]. In this line of work, a regressor maps aggregate detection statistics to a target count. The regressor is usually tested only on the detector outputs at hand, so detector quality and counter quality remain coupled.

What is missing is a benchmark that determines, for tree-level multi-view BBC counting, whether the limiting factor lies in the detector or in the tree-level counter. Existing reports usually fix one component and tune the other, which conflates two error sources. This benchmark tests that gap on SawitMVC by evaluating the same counters under GT detections and cached YOLO26m outputs, ablating eight feature banks under realistic detector outputs, and measuring how much of the 20.57 percentage-point Class ±1 Acc gap can be recovered by feature richness or counter choice.

## II. Methodology

### A. Dataset and Task Definition

The benchmark uses the SawitMVC multi-view dataset: 953 oil palm trees, 3,992 side-view images, four maturity classes (B1, B2, B3, and B4), and 4 to 8 side views per tree. The fixed split contains 716 training trees, 96 validation trees, and 141 test trees. Dataset construction, annotation protocol, and per-class composition are documented in the Zenodo dataset release (doi: 10.5281/zenodo.20336323).

For tree $i$, the ground-truth target is the count vector

$$
\mathbf{y}_i = [\,y_{i,B1},\;y_{i,B2},\;y_{i,B3},\;y_{i,B4}\,],
$$

and the prediction is $\hat{\mathbf{y}}_i = [\,\hat{y}_{i,B1},\ldots,\hat{y}_{i,B4}\,]$. The evaluation unit is the tree, not the individual image: per-image appearance counts are not valid BBC outputs.

Let $a_{i,c}$ denote the number of detected appearances of class $c$ across all views of tree $i$. The naive multi-view estimator

$$
\hat{y}_{i,c}^{\text{naive}} = a_{i,c}
$$

is biased upward whenever a bunch is visible from more than one side. Under GT annotations, this estimator reaches only 50.00% Class ±1 Acc and 6.38% Tree ±1 Acc on the test split, which motivates learned tree-level aggregation rather than direct summation.

### B. Detection Conditions

The benchmark isolates detector error from counter error by evaluating the same counters under two detection conditions, illustrated in Fig. 2.

![GT detection and fixed-detector pipeline diagram](figures/paper/fig02_detection_conditions.png)

**Fig. 2.** Two detection conditions used in the benchmark. *Top:* the GT detection setting supplies ground-truth boxes and classes to the tree-level feature extractor, so detector output quality is removed as an error source. *Bottom:* the fixed-detector setting supplies cached YOLO26m outputs, so detector misses and class confusions enter the pipeline before counting.

In the *GT detection setting*, the feature extractor reads ground-truth bounding boxes and classes directly from the annotation files. Counters trained on these features upper-bound how well a tree-level counter can perform when detections are accurate. In the *fixed-detector setting*, the feature extractor reads cached YOLO26m [15] predictions from the released `y26mv2` checkpoint. The detector was trained on the official 716-tree training split for 60 epochs (batch size 32, image size 640, patience 60, seed 42). Inference uses the repository default confidence threshold of 0.25 and the standard Ultralytics post-processing settings. A single detector checkpoint is used for all counter and feature configurations, so any observed differences come from the counter or feature representation rather than detector retraining.

**Table 1. YOLO26m validation detection performance, reported under the COCO mAP50 convention [16].**

| Class | mAP50 | Recall |
|:---:|---:|---:|
| B1 | **0.746** | **0.801** |
| B2 | 0.425 | 0.433 |
| B3 | 0.550 | 0.656 |
| B4 | 0.363 | 0.389 |
| Overall | 0.521 | 0.570 |

The detector is strongest on B1 and weakest on B4. The per-class weakness on B4 and the moderate recall on B3 visible in Table 1 are revisited in Section III-D, where the per-class structure of the fixed-detector gap mirrors this profile.

### C. Feature Vectors

For each tree, detections from all views are aggregated into a fixed-length feature vector. The 13-dimensional baseline F0 contains, for each class $c$, the total count $s_c$, the per-side maximum $m_c$, and the per-side mean $\mu_c$, plus the number of available views $n_{\text{views}}$:

$$
F0 = [\,s_{B1{:}B4},\;m_{B1{:}B4},\;\mu_{B1{:}B4},\;n_{\text{views}}\,].
$$

Three optional groups extend F0. The confidence group (20-dim) stores five statistics per class: confidence sum, mean, max, count above 0.5, and count above 0.6. The spatial group (8-dim) stores the mean normalized vertical centroid $\overline{cy}_c$ and mean bounding-box area $\overline{A}_c$ for each class, reflecting the vertical separation of maturity classes on the tree. The side-distribution group (20-dim) stores per-side std, per-side min, coefficient of variation, number of sides with detections, and a consistency score for each class.

A cross-class composition group (6-dim), consisting of the total detection count, four class fractions, and a B3-vs-(B2+B3) mixture ratio, completes the 67-dimensional combined bank F<sub>all</sub>:

$$
F_{\text{all}} = F0 \cup \text{conf} \cup \text{spatial} \cup \text{distrib} \cup \text{composition}.
$$

Eight feature banks are evaluated: F0, F0+conf, F0+spatial, F0+distrib, F0+conf+spatial, F0+conf+distrib, F0+distrib+spatial, and F<sub>all</sub>.

### D. Counter Models and Evaluation Metrics

Five regression models are evaluated in both detection conditions: Linear Regression, Ridge [19], ElasticNet [17], SVM [20], and Random Forest [18]. Each counter maps a tree feature vector $\mathbf{x}_i$ to the count vector $\hat{\mathbf{y}}_i = f_{\theta}(\mathbf{x}_i)$.

All counter models are trained on the 716-tree training split and evaluated on the fixed 141-tree test split. Class-level $\pm 1$ correctness for tree $i$ and class $c$ is

$$
I_{i,c}^{\pm 1} =
\begin{cases}
1, & |\hat{y}_{i,c} - y_{i,c}| \leq 1, \\
0, & \text{otherwise}.
\end{cases}
$$

**Class ±1 Acc** averages this indicator over all test trees and classes:

$$
\text{Class } \pm 1 \text{ Acc} = \frac{1}{4N}\sum_{i=1}^{N}\sum_{c=1}^{4} I_{i,c}^{\pm 1}.
$$

**Tree ±1 Acc** is stricter: a tree counts as correct only if all four class predictions are within $\pm 1$ simultaneously,

$$
\text{Tree } \pm 1 \text{ Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left(\sum_{c=1}^{4} I_{i,c}^{\pm 1} = 4\right).
$$

**Macro MAE** is the mean absolute error averaged over trees and classes. **Signed class bias** is the per-class mean $\hat{y}_{i,c} - y_{i,c}$; positive values indicate over-counting, negative values under-counting. Bias is reported alongside accuracy because operational aggregate yield estimates are sensitive to directional error rather than absolute error alone.

## III. Results and Discussion

The results follow the detector-counter question directly. Section III-A estimates the counter ceiling under GT detections, Section III-B measures the deployed fixed-detector setting, Section III-C tests whether richer features recover the loss, and Section III-D compares the two conditions per class.

### A. GT Detection Setting

Table 2 reports counting results when the counter receives GT detections. Naive appearance summation reaches only 50.00% Class ±1 Acc and 6.38% Tree ±1 Acc, confirming that duplicate visibility cannot be ignored. The two divisor rules recover most of the loss, and ElasticNet lifts the result to 98.05% Class ±1 Acc. Under accurate detections, the remaining counting problem is therefore simple enough for compact regression models.

**Table 2. Counting under the GT detection setting on the 141-tree test split.**

| Method | Type | Class ±1 | Tree ±1 | MAE |
|---|---|---:|---:|---:|
| Naive sum | Heur. | 50.00% | 6.38% | 2.142 |
| Global divisor | Heur. | 95.39% | 85.11% | 0.376 |
| Visibility-adapt. divisor | Heur. | 95.92% | 87.23% | 0.340 |
| ElasticNet | ML | **98.05%** | **92.20%** | 0.277 |
| SVM | ML | 97.87% | 91.49% | **0.266** |
| Ridge | ML | 97.70% | 90.78% | 0.275 |
| Linear reg. | ML | 97.52% | 90.07% | 0.277 |
| Random forest | ML | 95.92% | 84.40% | 0.365 |

The four best machine-learning counters (ElasticNet, SVM, Ridge, LR) sit within 0.5 percentage points of each other on Class ±1 Acc and within 2.2 points on Tree ±1 Acc. Random Forest is the only counter that underperforms in this setting. This narrow spread indicates that, when detector evidence is accurate, the choice among standard counter families is not the limiting factor.

### B. Fixed-Detector Setting

Table 3 holds the feature bank at F0 and compares the five counter families when they receive cached YOLO26m outputs. The best F0 result, ElasticNet at 76.42% Class ±1 Acc, trails the same counter under GT detections by more than 20 percentage points. All five counters remain within a 3.2 percentage-point window on Class ±1 Acc. The large drop across the two detection conditions, coupled with the small spread among counters, points to the detector output rather than the counter family as the main source of loss in this fixed-detector setting.

**Table 3. Fixed-detector counting with F0 features on the 141-tree test split.**

| Counter | Class ±1 Acc | Tree ±1 Acc | Macro MAE |
|---|---:|---:|---:|
| ElasticNet | **76.42%** | 29.79% | **1.043** |
| Ridge | 76.06% | 28.37% | 1.053 |
| Linear Regression | 75.71% | **30.50%** | 1.048 |
| SVM | 74.82% | 29.08% | **1.043** |
| Random Forest | 73.23% | 26.95% | 1.110 |

Per-class accuracies in Table 4 locate the failure mode. B1 stays above 95% across all five counters, B2 remains between 73.76% and 80.14%, and B3 falls to roughly 55-56%. B4 is intermediate, between 67% and 73%. This profile mirrors Table 1: the classes with weaker detection recall are the same classes that lose the most tree-level counting accuracy.

**Table 4. Per-class Class ±1 Acc of fixed-detector counters with F0 features on the 141-tree test split.**

| Counter | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|
| ElasticNet | **96.45%** | **80.14%** | **56.03%** | **73.05%** |
| Ridge | 95.74% | **80.14%** | **56.03%** | 72.34% |
| Linear Regression | **96.45%** | 79.43% | 55.32% | 71.63% |
| SVM | 95.74% | 76.60% | 55.32% | 71.63% |
| Random Forest | 95.74% | 73.76% | **56.03%** | 67.38% |

### C. Feature Ablation

**Table 5. Ridge feature ablation in the fixed-detector setting.**

| Features | Dim | Class ±1 | Tree ±1 | MAE |
|---|---:|---:|---:|---:|
| F0 | 13 | 76.06% | 28.37% | 1.053 |
| F0 + confidence | 33 | 75.89% | 29.79% | 1.059 |
| F0 + spatial | 21 | 76.60% | 30.50% | 1.046 |
| F0 + side-distribution | 33 | 74.82% | 28.37% | 1.060 |
| F0 + confidence + spatial | 41 | 76.24% | 29.08% | 1.059 |
| F0 + confidence + side-distribution | 53 | 76.42% | 29.79% | 1.060 |
| F0 + side-distribution + spatial | 41 | 75.71% | 30.50% | 1.048 |
| F<sub>all</sub> | 67 | **77.48%** | **32.62%** | **1.035** |

Table 5 holds the counter family fixed at Ridge, the strongest counter under the richest feature bank, and varies only the feature configuration. F0 starts at 76.06% Class ±1 Acc. Adding spatial features alone raises this to 76.60%, while confidence and side-distribution features do not improve the baseline by themselves.

The full F<sub>all</sub> bank gives the best ablation result at 77.48% Class ±1 Acc and 32.62% Tree ±1 Acc. The gain over F0 is 1.42 percentage points at class level and 4.25 points at tree level. This is the largest improvement produced by feature ablation, but it recovers only a small fraction of the 20.57 percentage points lost between GT and fixed-detector inputs.

### D. GT vs Fixed-Detector Gap

The central comparison is the per-class gap between ideal detections and the fixed detector. Fig. 3 plots this gap in the top panel and shows the signed bias of the best fixed-detector configuration in the bottom panel.

![Per-class GT-vs-fixed-detector accuracy gap and fixed-detector bias](figures/paper/fig03_gap_bias.png)

**Fig. 3.** *Top:* Per-class Class ±1 Acc for the best counter in each detection condition (ElasticNet under GT; Ridge + F<sub>all</sub> under the fixed detector). Counting is near ceiling under GT detection but drops sharply under the fixed detector. *Bottom:* Per-class signed bias of Ridge + F<sub>all</sub> in the fixed-detector setting. Negative values indicate systematic under-counting; the under-count is concentrated on B2 and especially B3.

The absolute gap is large, at 20.57 percentage points in Class ±1 Acc and 59.58 percentage points in Tree ±1 Acc, as summarized in Table 6. The loss is not uniform across maturity classes. B1 loses only ≈2.8 pp between the two conditions, while B2 loses ≈16.3 pp, B3 loses 36.2 pp, and B4 loses ≈27.0 pp. This profile aligns with Table 1, where B3 and B4 have weaker detector recall and fall furthest below the GT ceiling.

The bias plot adds direction to the accuracy gap. The fixed-detector Ridge + F<sub>all</sub> pipeline under-counts B2 by 0.078 and B3 by 0.177 bunches per tree on average. Because plantation-level forecasts aggregate per-tree predictions across blocks, systematic under-counting in B2 and B3 would carry into the next-cycle harvest estimate rather than cancel out as random noise.

**Table 6. Consolidated test-set summary. Biases are ordered by class.**

| Config | Class ±1 | Tree ±1 | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|---:|---:|
| GT, ElasticNet | 98.05% | 92.20% | −0.050 | +0.043 | −0.064 | −0.028 |
| Fixed, Ridge + F<sub>all</sub> | 77.48% | 32.62% | +0.014 | −0.078 | −0.177 | +0.071 |

Read together, Tables 1 through 6 with Fig. 3 support a single interpretation. Multi-view duplicate visibility is real and large, but learned tree-level counters handle it when detector evidence is accurate (Section III-A). Once a fixed detector replaces ground-truth boxes, counter performance drops sharply (Section III-B), and richer features or different counter families recover only a small fraction of the loss (Section III-C). The per-class structure of the gap and bias is consistent with a detector-quality bottleneck under the tested YOLO26m checkpoint.

Several limitations bound the scope of these findings. The benchmark uses a single dataset of 953 trees from two plantations in South Kalimantan, so absolute numbers may shift under other regions, varieties, or acquisition protocols. Only one detector checkpoint (YOLO26m) is tested, and a different detector family could change the per-class error profile. The tree-level counters treat each tree independently and ignore plantation-scale spatial or temporal regularities that operational forecasts sometimes exploit. The four-class B1 through B4 taxonomy also admits visual ambiguity at the B2/B3 and B3/B4 boundaries. Future work should prioritize B3/B4 recall, confidence calibration for counting, and explicit cross-view association or geometry-aware aggregation [2], [3], [4] rather than expanding the tree-level feature bank further.

## IV. Conclusion

This paper benchmarks multi-view tree-level oil palm FFB counting by evaluating the same counters under GT detections and cached YOLO26m outputs. With GT detections, standard regression counters recover the B1 to B4 count with near-ceiling accuracy. With the fixed detector, the best Ridge + F<sub>all</sub> pipeline reaches 77.48% Class ±1 Acc, and feature ablation recovers only a small fraction of the gap. On SawitMVC under this detector checkpoint, the limiting factor is detector output quality, especially for B3 and B4, rather than the choice among the tested tree-level counters.

## Acknowledgment

This research was funded by the Badan Pengelola Dana Perkebunan (BPDP), Indonesia, under contract number PRJ-36/BPDP/2026, through Universitas Lambung Mangkurat (ULM), Banjarmasin, Indonesia. The authors thank the field data collection teams and annotators whose work produced the bounding box and cross-view identity labels. The authors also thank the plantation staff for supervising annotation and the plantation management for facilitating field access.

## Code and Data Availability

Code: https://github.com/ULM-SawitMVC/Baseline-SawitMVC

Data: SawitMVC is available on Zenodo at https://zenodo.org/records/20336323, doi: 10.5281/zenodo.20336323.

## References

[1] A. Koirala, K. B. Walsh, Z. Wang, and C. McCarthy, "Deep learning: method overview and review of use for fruit detection and yield estimation," *Computers and Electronics in Agriculture*, vol. 162, pp. 219-234, Jul. 2019, doi: 10.1016/j.compag.2019.04.017.

[2] X. Liu, S. W. Chen, C. Liu, S. S. Shivakumar, J. Das, C. J. Taylor, J. Underwood, and V. Kumar, "Monocular camera based fruit counting and mapping with semantic data association," *IEEE Robotics and Automation Letters*, vol. 4, no. 3, pp. 2296-2303, Jul. 2019, doi: 10.1109/LRA.2019.2901987.

[3] M. Gaillard, B. Benes, M. C. Tross, and J. C. Schnable, "Multi-view triangulation without correspondences," *Computers and Electronics in Agriculture*, vol. 206, Art. no. 107688, Mar. 2023, doi: 10.1016/j.compag.2023.107688.

[4] H. Freeman, E. Schneider, C. H. Kim, M. Lee, and G. Kantor, "3D reconstruction-based seed counting of sorghum panicles for agricultural inspection," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, 2023, pp. 9594-9600, doi: 10.1109/ICRA48891.2023.10161400.

[5] M. Stein, S. Bargoti, and J. Underwood, "Image based mango fruit detection, localisation and yield estimation using multiple view geometry," *Sensors*, vol. 16, no. 11, Art. no. 1915, Nov. 2016, doi: 10.3390/s16111915.

[6] P. Roy, A. Kislay, P. A. Plonski, J. Luby, and V. Isler, "Vision-based preharvest yield mapping for apple orchards," *Computers and Electronics in Agriculture*, vol. 164, Art. no. 104897, Sep. 2019, doi: 10.1016/j.compag.2019.104897.

[7] N. Häni, P. Roy, and V. Isler, "A comparative study of fruit detection and counting methods for yield mapping in apple orchards," *Journal of Field Robotics*, vol. 37, no. 2, pp. 263-282, Mar. 2020, doi: 10.1002/rob.21902.

[8] L. Meyer, A.-T. Ardelean, T. Weyrich, and M. Stamminger, "FruitNeRF++: A generalized multi-fruit counting method utilizing contrastive learning and neural radiance fields," in *Proc. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2025, pp. 1717-1724, doi: 10.1109/IROS60139.2025.11247341.

[9] N. A. Prasetyo, Pranowo, and A. J. Santoso, "Automatic detection and calculation of palm oil fresh fruit bunches using Faster R-CNN," *International Journal of Applied Science and Engineering*, vol. 17, no. 2, pp. 121-134, 2020, doi: 10.6703/IJASE.202005_17(2).121. [Online]. Available: https://gigvvy.com/journals/ijase/articles/ijase-202005-17-2-121

[10] J. W. Lai, H. R. Ramli, L. I. Ismail, and W. Z. W. Hasan, "Real-time detection of ripe oil palm fresh fruit bunch based on YOLOv4," *IEEE Access*, vol. 10, pp. 95763-95770, 2022, doi: 10.1109/ACCESS.2022.3204762.

[11] M. Y. M. A. Mansour, K. D. Dambul, and K. Y. Choo, "Object detection algorithms for ripeness classification of oil palm fresh fruit bunch," *International Journal of Technology*, vol. 13, no. 6, pp. 1326-1335, 2022, doi: 10.14716/ijtech.v13i6.5932.

[12] M. Kerstan and T. Fairhurst, "Use of OMP for accurate crop forecasts," Agrisoft Systems and Tropical Crop Consultants Limited. Available: https://www.agrisoft-systems.com/wp-content/uploads/Papers/CropForecastsWithOMPBBC.pdf

[13] M. H. Junos, A. S. M. Khairuddin, S. Thannirmalai, and M. Dahari, "Automatic detection of oil palm fruits from UAV images using an improved YOLO model," *The Visual Computer*, vol. 38, no. 7, pp. 2341-2355, 2022, doi: 10.1007/s00371-021-02116-3.

[14] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 7464-7475, doi: 10.1109/CVPR52729.2023.00721.

[15] G. Jocher and J. Qiu, "Ultralytics YOLO26," version 26.0.0, 2026. [Software]. Available: https://github.com/ultralytics/ultralytics. License: AGPL-3.0.

[16] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in *Proc. European Conference on Computer Vision (ECCV)*, 2014, pp. 740-755, doi: 10.1007/978-3-319-10602-1_48.

[17] H. Zou and T. Hastie, "Regularization and variable selection via the elastic net," *Journal of the Royal Statistical Society B*, vol. 67, no. 2, pp. 301-320, 2005, doi: 10.1111/j.1467-9868.2005.00503.x.

[18] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.

[19] A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased estimation for nonorthogonal problems," *Technometrics*, vol. 12, no. 1, pp. 55-67, 1970, doi: 10.1080/00401706.1970.10488634.

[20] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273-297, 1995, doi: 10.1007/BF00994018.
