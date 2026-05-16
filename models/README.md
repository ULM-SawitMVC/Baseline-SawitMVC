# Models — YOLO26 Detection Weights

Five trained YOLO26 weights for oil palm bunch detection across B1–B4 maturity classes.
All models were trained on `Brand-New-Dataset-YOLO` (3,992 images, 4 classes).

---

## Model Comparison

| File | Architecture | mAP50 | mAP50-95 | Speed | Size | Training |
|------|-------------|:-----:|:--------:|:-----:|:----:|----------|
| `y26n_vanilla_local.pt` | YOLO26n | **0.521** | **0.237** | **0.2 ms** | 5.2 MB | Pretrained + aug ✓ |
| `y26s_nopretrained.pt` | YOLO26s | 0.511 | 0.231 | 0.5 ms | 20 MB | Scratch + aug ✓ |
| `y26m_vanilla_local.pt` | YOLO26m | 0.509 | 0.231 | 0.8 ms | 42 MB | Pretrained + aug ✓ |
| `y26s_vanilla_local.pt` | YOLO26s | 0.506 | 0.235 | 0.5 ms | 20 MB | Pretrained + aug ✓ |
| `y26s_noaug.pt` | YOLO26s | 0.465 | 0.216 | 0.5 ms | 20 MB | Pretrained, no aug ✗ |

**Recommended for production:** `y26n_vanilla_local.pt` — highest mAP50 with the
fastest inference. The nano model outperforms the medium model while running 4× faster.

**Best for E2E pipeline:** `y26m_vanilla_local.pt` — achieves the best combined
Acc±1 (71.6%) when paired with an SVM counter. See `benchmarks/e2e/`.

---

## Key Findings

1. **Augmentation is critical** — `y26s_noaug` drops 0.056 mAP50 and overfits at epoch 6.
   Always use default Ultralytics augmentations.

2. **Pretraining is optional** — `y26s_nopretrained` (scratch, mAP50=0.511) slightly
   outperforms `y26s_vanilla_local` (pretrained, mAP50=0.506). COCO weights are not
   a strict requirement for this domain.

3. **Nano beats medium** — `y26n` achieves the highest mAP50 (0.521) at 4× the speed
   of `y26m`. For this dataset, smaller architecture generalizes better.

4. **Detector is the bottleneck** — Even the best model (y26n) produces only ~71% E2E
   Acc±1 when paired with a counting algorithm, versus 87.6% when using ground-truth
   detections. Improving detection quality is the highest-leverage improvement.

---

## Training Configuration

All vanilla models used the same hyperparameters:

```yaml
model:      yolo26{n,s,m}.pt  # COCO pretrained
data:       Brand-New-Dataset-YOLO/data.yaml
imgsz:      640
batch:      16
epochs:     100
patience:   50
seed:       42
deterministic: true
optimizer:  auto              # Ultralytics default
```

Augmentation (default Ultralytics): HSV shift, horizontal flip, translate, scale,
mosaic, copy-paste. See `docs/training.md` for full reproduction guide.

---

## Inference Code

### Detect bunches in a single image

```python
from ultralytics import YOLO

model = YOLO("models/y26n_vanilla_local.pt")
results = model.predict("path/to/image.jpg", conf=0.25, iou=0.45)

for r in results:
    for box in r.boxes:
        cls_name = r.names[int(box.cls)]  # "B1", "B2", "B3", or "B4"
        x_center, y_center, w, h = box.xywhn[0].tolist()  # normalized
        print(f"{cls_name}: x={x_center:.3f}, y={y_center:.3f}")
```

### Run detections on a full tree (4 images) and count

```python
from pathlib import Path
from ultralytics import YOLO
from algorithms.M01_selector_b2b3 import predict

model = YOLO("models/y26n_vanilla_local.pt")

# Assume images are named: TREEID_1.jpg, TREEID_2.jpg, TREEID_3.jpg, TREEID_4.jpg
image_paths = sorted(Path("SawitMVC-YOLO/images/").glob("DAMIMAS_A21B_0001_*.jpg"))

detections = []
for side_index, img_path in enumerate(image_paths):
    results = model.predict(img_path, conf=0.25, iou=0.45, verbose=False)
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "x_norm": float(box.xywhn[0][0]),
                "y_norm": float(box.xywhn[0][1]),
                "side_index": side_index,
            })

count = predict(detections)
print(count)  # {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
```

### Evaluate on the full test set

```bash
yolo detect val model=models/y26n_vanilla_local.pt data=SawitMVC-YOLO/data.yaml split=test
```

---

## Training Artifacts

Training curves, confusion matrices, and P-R curves for all 5 models are available in:

```
figures/training/
├── y26n_vanilla_local/     confusion_matrix.png, results.png, BoxPR_curve.png, ...
├── y26s_vanilla_local/
├── y26m_vanilla_local/
├── y26s_nopretrained/
└── y26s_noaug/
```

See `figures/README.md` for a full index.
