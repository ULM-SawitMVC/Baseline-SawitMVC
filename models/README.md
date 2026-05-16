# Models — YOLO26 Detection Weights

Three trained YOLO26 weights for oil palm bunch detection across B1–B4 maturity classes.
All models trained on `SawitMVC-YOLO` (3,992 images, 4 classes), 60 epochs, seed=42.

---

## Model Comparison

| File | Architecture | mAP50 | mAP50-95 | Speed | Size | Notes |
|------|-------------|:-----:|:--------:|:-----:|:----:|-------|
| `y26m.pt` | YOLO26m | **0.528** | **0.243** | 1.0 ms | 42 MB | Best detection accuracy |
| `y26n.pt` | YOLO26n | 0.515 | 0.232 | **0.3 ms** | **5.2 MB** | ⚡ Best efficiency |
| `y26s.pt` | YOLO26s | 0.511 | 0.239 | 0.4 ms | 20 MB | Balanced size/speed |

**Recommended for production:** `y26n.pt` — near-best mAP50 at the lowest cost.

**Best E2E Acc±1:** `y26n.pt` paired with M01 achieves **74.4%** — see `benchmarks/e2e/`.

---

## Training Configuration

All 3 models share identical hyperparameters (only architecture differs):

```yaml
model:      yolo26{n,s,m}  # COCO pretrained weights
data:       ul://sawit-ulm/datasets/sawitmvc-yolo
imgsz:      640
batch:      32
epochs:     60
patience:   60
seed:       42
deterministic: true
optimizer:  auto   # AdamW, lr=0.00125
```

Full training logs per model: `models/y26n_train_log.txt`, `y26s_train_log.txt`, `y26m_train_log.txt`.

---

## Per-Class Validation Results (best.pt)

| Model | B1 mAP50 | B2 mAP50 | B3 mAP50 | B4 mAP50 | All mAP50 |
|-------|:--------:|:--------:|:--------:|:--------:|:---------:|
| y26n  | 0.740 | 0.373 | 0.577 | 0.369 | **0.515** |
| y26s  | 0.720 | 0.427 | 0.581 | 0.317 | **0.511** |
| y26m  | 0.788 | 0.395 | 0.602 | 0.330 | **0.528** |

B1 (ripe red bunches) is the easiest class; B4 (newest, dark green) is the hardest.

---

## Inference Code

### Detect bunches in a single image

```python
from ultralytics import YOLO

model = YOLO("models/y26n.pt")
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

model = YOLO("models/y26n.pt")

# Images named: TREEID_1.jpg, TREEID_2.jpg, TREEID_3.jpg, TREEID_4.jpg
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

### Evaluate on the val split

```bash
yolo detect val model=models/y26n.pt data=SawitMVC-YOLO/data.yaml split=val
```
