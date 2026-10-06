# Presentation Script: ICIC 2026

**Paper Title:** Benchmarking Multi-View Tree-Level Oil Palm Bunch Counting Under a Fixed Detector
**Presenter:** Muhammad Zainal Muttaqin
**Target Duration:** about 4 minutes (635 words)
**Slide Deck:** `057_MUTTAQIN.pptx` (7 slides, 4 sections)
**How to read:** The slides carry the numbers. The script explains each idea in plain words and introduces every term before it is used. *[Brackets]* mark an action, not speech.

---

### Slide 1: Title

Good morning, everyone. Today I will present our paper, titled "Benchmarking Multi-View Tree-Level Oil Palm Bunch Counting Under a Fixed Detector". I am Muhammad Zainal Muttaqin from Lambung Mangkurat University, and I am presenting this work on behalf of my co-authors.

---

### Slide 2: Problem and research objective (Section 1, Introduction)

In an oil palm plantation, the manager needs to know how many bunches are on each tree and how ripe they are, in four maturity classes from B1 to B4. Today workers count them by eye, so we want to do it from photos.

One photo cannot see the whole tree, so we take four photos from four sides. The problem is that the same bunch then appears in several photos, almost twice on average, so we cannot simply add up what we see in each photo. *[point: orange boxes]*

A system for this has two parts. A detector finds the bunches in each photo and draws a box around each one. A counter then takes the boxes from all the photos and gives the final number of bunches for the tree. Our research objective is to find out which part causes the counting error, the detector or the counter.

---

### Slide 3: Related work and research gap (Section 2, Background)

Other researchers have solved this for mango and apple, but their methods need a camera that moves along the row or many overlapping images. We only have four separate smartphone photos per tree.

For oil palm, earlier work stops at detecting bunches in a single image. And when a counting system is tested, it is tested as a whole, so nobody can tell which of the two parts made the mistake.

---

### Slide 4: Experimental design (Section 2, Background)

To separate the two parts, we test every counter twice. The first time, we give it the boxes that experts drew by hand, so the input has no mistakes. *[point: top image]* The second time, we give it the boxes from a YOLO detector, which misses some bunches and gives some of them the wrong class. *[point: bottom image]*

We use one detector model for the whole study and we never retrain it, so every counter receives exactly the same boxes. Because the counter is also the same in both tests, any drop in accuracy comes from the mistakes of the detector.

As counters, we compare two simple rules, such as dividing the total by the average number of repeats, and five regression models. We count a prediction as correct when it is within one bunch of the true number.

---

### Slide 5: Results on the 141 test trees (Section 3, Results)

This slide has all our results, so let me point to the main numbers. With the expert boxes, the best counter is correct in 98 percent of the cases. With the detector boxes, it drops to 77 percent. That is a loss of about 21 points. *[point: top tiles]*

In contrast, when we compare the five counters with each other, they differ by only 3 points, and that difference is not statistically significant.

Why is the loss so large? Because 23 percent of the bunches are not detected in any of the photos, and the counter cannot count a bunch that it never receives. The detector also often labels B2 bunches as B3. *[point: panels C and D]*

Giving the counter more information improved it by only about one point. And when we repeated the comparison with another detector and with cross-validation, the gap stayed between 23 and 25 points.

---

### Slide 6: Conclusions (Section 4, Conclusion)

So, to answer our question: on this dataset, the count depends much more on the detector than on the choice of counter. The next improvement should therefore be in the detector, especially for the unripe classes B3 and B4.

We also suggest that counting methods be reported with both expert boxes and detector boxes, so that the two sources of error stay visible.

Our data come from only two estates, so we still need to test this on an independent plantation.

---

### Slide 7: Thank you

Thank you for listening. Our code and results are on GitHub, and I am happy to take your questions.

---

## Anticipated Questions & Answers (Cheat Sheet)

**Q: What exactly are the features? Can you list them?**  
**A:** Sure. The baseline has thirteen features. For each of the four classes we take three numbers: the total detections over all views, the maximum in a single view, and the mean per view. That is twelve, and the last one is the number of views. The extended set has 67 features. On top of the baseline, it adds four groups. The first is confidence: the sum, mean and maximum confidence for each class, and how many boxes are above 0.5 and above 0.6. The second is position and size: the average vertical position of the boxes and their average area. The third is the spread across views: the standard deviation, the minimum, the coefficient of variation, how many views contain the class, and a consistency score. And the last one is composition: the total number of detections, the share of each class, and the share of B3 among B2 and B3.

*(Quick count if asked: 13 baseline + 20 confidence + 8 position and size + 20 spread + 6 composition = 67.)*

**Q: Is harvest planning in real plantations based only on the Black Bunch Census?**  
**A:** No, not only that. The daily harvest is decided by the harvesters in the field, who check the ripe bunches and the loose fruit on each round, typically every 7 to 10 days. The Black Bunch Census is a periodic survey, usually every three or four months, and it is mainly used to forecast the crop one to four months ahead and to plan labor and transport. Our work is aimed at automating that census.

**Q: Why not use 3D reconstruction or tracking to match bunches across views?**  
**A:** That is a good question. Those methods usually need a lot of overlap between images, a moving sensor, or calibrated cameras. In our case we only have four to eight smartphone photos per tree, taken from different sides. So we started with a simpler approach, which is regression on summary features. Comparing it with explicit matching on the same inputs is the next thing we want to do.

**Q: Is ground truth an upper bound for counting accuracy?**  
**A:** Not really. We treat it as an empirical reference. It shows what these counters can reach when the boxes and classes are all correct. A different counter in the future could use information that ours do not use, so we do not want to call it an upper bound.

**Q: Why does Random Forest perform worse than the linear counters?**  
**A:** A random forest makes its prediction by averaging similar training examples, so it tends to pull high counts down towards the more common values. For example, for B1 it never predicts more than three bunches, even though some trees have five or six. I should add that with detector boxes, this difference is no longer statistically significant.

**Q: Would a larger or newer detector close the gap?**  
**A:** Not in the experiments we did. With YOLO11m the gap is about 24 points, and with four YOLO26 checkpoints it is about 23. But I have to be careful here, because these are all YOLO models at a similar level, around 0.52 mAP50 for YOLO26m, and YOLO11m was only trained once. A much stronger detector should make the gap smaller, and that is exactly where we suggest putting the effort.

**Q: Why not lower the confidence threshold to recover the missed bunches?**  
**A:** We tried that. We tested eleven thresholds from 0.05 to 0.70. A lower threshold does find more bunches, but it also adds a lot of false boxes, and none of the thresholds gives more than 77.5 percent at class level. At 0.10 the tree-level accuracy goes up from 33 to 38 percent, but we ran this on the test set, so we only report it as an exploratory result.

**Q: Why is B3 the hardest class to count, even though its recall is higher than B2 and B4?**  
**A:** There are two reasons. First, B3 is the largest class, more than half of all bunch appearances in the test set, so a tree usually has more B3 bunches and it is easier to be off by two. Second, B3 also receives the B2 bunches that the detector labels wrongly. So recall alone does not tell us which class is hardest to count.

**Q: Why a tolerance of one bunch? What about other metrics?**  
**A:** The counts per class are small, around two to three bunches per tree on average, so one bunch is the smallest tolerance after an exact match. In the paper we also report MAE, RMSE and bias, and they show the same pattern. The mean error per class goes from about 0.3 bunches with ground truth boxes to about 1.0 with detector boxes.

**Q: What exactly is the "fixed detector", and why did you keep it fixed?**  
**A:** It is a single YOLO26m model that we trained once on the 716 training trees and then froze, with a confidence threshold of 0.25. Because it is fixed, every counter receives exactly the same boxes. So if two counters give different results, we know the difference does not come from the detector.

**Q: How well do these results generalize to other plantations?**  
**A:** We cannot claim that yet. The data come from two estates in South Kalimantan, and the smaller one only has 99 trees. In our cross-estate test, the effect of the estate is mixed with the effect of the training set size, and the detector has seen both estates. So testing on an independent plantation is part of our future work.
