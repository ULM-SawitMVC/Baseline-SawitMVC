# Models

This folder contains every model artifact in the repository, split into two
families.

| Subfolder | Contents |
|-----------|----------|
| [`yolo/`](yolo/) | Three YOLOv26 detector weights with per-epoch training logs. |
| [`counters/`](counters/) | Three saved scikit-learn counters used by Track B (SVM, RF, LR). See [`counters/README.md`](counters/README.md). |

## YOLOv26 detectors

All three weights were retrained on the bundled SawitMVC-YOLO release
(3,992 images, four classes), with `seed=42`, `imgsz=640`, sixty epochs.

| File | Architecture | mAP50 | mAP50–95 | Inference speed | Size | Recommended use |
|------|--------------|------:|---------:|----------------:|-----:|-----------------|
| [`yolo/y26m.pt`](yolo/y26m.pt) | YOLOv26 medium | **0.528** | **0.243** | 1.0 ms | 42 MB | Highest detection accuracy. |
| [`yolo/y26n.pt`](yolo/y26n.pt) | YOLOv26 nano | 0.515 | 0.232 | **0.3 ms** | **5.2 MB** | Best efficiency; recommended for deployment. |
| [`yolo/y26s.pt`](yolo/y26s.pt) | YOLOv26 small | 0.511 | 0.239 | 0.4 ms | 20 MB | Strongest paired with the SVM counter (70.79% E2E Acc±1). |

Per-epoch curves, optimizer choice, and exact CLI invocations are captured in:

- [`yolo/train_logs/y26n_train_log.txt`](yolo/train_logs/y26n_train_log.txt)
- [`yolo/train_logs/y26s_train_log.txt`](yolo/train_logs/y26s_train_log.txt)
- [`yolo/train_logs/y26m_train_log.txt`](yolo/train_logs/y26m_train_log.txt)

### Training configuration

```yaml
model:         yolo26{n,s,m}  # COCO pretrained
data:          SawitMVC-YOLO/data.yaml
imgsz:         640
batch:         32
epochs:        60
patience:      60
seed:          42
deterministic: true
optimizer:     auto      # AdamW, lr=0.00125
```

### Per-class validation mAP50

| Model | B1 | B2 | B3 | B4 | All |
|-------|---:|---:|---:|---:|----:|
| y26n  | 0.740 | 0.373 | 0.577 | 0.369 | **0.515** |
| y26s  | 0.720 | 0.427 | 0.581 | 0.317 | **0.511** |
| y26m  | 0.788 | 0.395 | 0.602 | 0.330 | **0.528** |

B1 (ripe red) is consistently the easiest class; B4 (small green-black) is
the hardest because of low contrast against the canopy.

## Inference usage

### Detect bunches in a single image

```python
from ultralytics import YOLO

model = YOLO("models/yolo/y26n.pt")
results = model.predict("path/to/image.jpg", conf=0.25, iou=0.45)

for r in results:
    for box in r.boxes:
        cls_name = r.names[int(box.cls)]                 # "B1" .. "B4"
        x_center, y_center, w, h = box.xywhn[0].tolist() # normalized
        print(f"{cls_name}: x={x_center:.3f}, y={y_center:.3f}")
```

### Detect and count a full tree (4 images)

```python
from pathlib import Path
from ultralytics import YOLO
from algorithms.M01_selector_b2b3 import predict

model = YOLO("models/yolo/y26n.pt")
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

print(predict(detections))   # e.g. {"B1": 1, "B2": 2, "B3": 5, "B4": 0}
```

### Validate against the held-out split

```bash
yolo detect val model=models/yolo/y26n.pt \
    data=ground_truth/data.yaml split=val
```

(Set `data:` to the SawitMVC-YOLO `data.yaml` path if you wish to evaluate on
the full image set.)

## Counter artifacts (Track B)

The SVM, Random Forest, and Linear Regression estimators used in Track B are
serialised under [`counters/`](counters/). They were fitted on
[`predictions/y26s_per_tree/`](../predictions/y26s_per_tree/) features and can
be reloaded with the `--load-model` flag of each
[`pipeline/run_counting_*.py`](../pipeline/) script. See
[`counters/README.md`](counters/README.md) for details and regeneration
commands.
