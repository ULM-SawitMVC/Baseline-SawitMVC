# Training — Reproduce YOLO26 Experiments

All five YOLO26 detection models can be reproduced using the commands and
configurations in this document. Training requires a GPU with ≥ 8 GB VRAM.

---

## Hardware Requirements

| Component | Minimum | Used in experiments |
|-----------|---------|---------------------|
| GPU | 8 GB VRAM | NVIDIA A40 (48 GB) |
| RAM | 16 GB | 128 GB |
| Storage | 5 GB (dataset) | NVMe SSD |
| Python | 3.10+ | 3.11 |

---

## Setup

```bash
pip install -r requirements.txt
# or specifically for training:
pip install ultralytics>=8.3.0 torch>=2.0.0
```

Download the dataset:
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ULM-DS-Lab/SawitMVC-YOLO', repo_type='dataset', local_dir='./SawitMVC-YOLO')
"
```

---

## Training Commands

All experiments use the same base hyperparameters unless stated otherwise.

### 1. YOLO26n — Vanilla (Recommended)

```bash
yolo detect train \
    model=yolo26n.pt \
    data=SawitMVC-YOLO/data.yaml \
    imgsz=640 \
    batch=16 \
    epochs=100 \
    patience=50 \
    seed=42 \
    deterministic=True \
    name=y26n_vanilla
```

Expected: **mAP50 ≈ 0.521** at best epoch ~38.

### 2. YOLO26s — Vanilla

```bash
yolo detect train \
    model=yolo26s.pt \
    data=SawitMVC-YOLO/data.yaml \
    imgsz=640 batch=16 epochs=100 patience=50 seed=42 deterministic=True \
    name=y26s_vanilla
```

Expected: **mAP50 ≈ 0.506** at best epoch ~21.

### 3. YOLO26m — Vanilla

```bash
yolo detect train \
    model=yolo26m.pt \
    data=SawitMVC-YOLO/data.yaml \
    imgsz=640 batch=16 epochs=100 patience=50 seed=42 deterministic=True \
    name=y26m_vanilla
```

Expected: **mAP50 ≈ 0.509** at best epoch ~33.

### 4. YOLO26s — No Pretrained Weights (Scratch)

```bash
yolo detect train \
    model=yolo26s.yaml \
    data=SawitMVC-YOLO/data.yaml \
    imgsz=640 batch=16 epochs=100 patience=50 seed=42 deterministic=True \
    name=y26s_nopretrained
```

Expected: **mAP50 ≈ 0.511** at best epoch ~57 (needs more epochs without head start).

### 5. YOLO26s — No Augmentation (Ablation)

```bash
yolo detect train \
    model=yolo26s.pt \
    data=SawitMVC-YOLO/data.yaml \
    imgsz=640 batch=16 epochs=100 patience=50 seed=42 deterministic=True \
    hsv_h=0 hsv_s=0 hsv_v=0 translate=0 scale=0 mosaic=0 \
    name=y26s_noaug
```

Expected: **mAP50 ≈ 0.465**, early stopping at epoch ~6 (overfitting).

---

## Hyperparameter Table

| Parameter | Vanilla | No-pretrained | No-aug |
|-----------|:-------:|:-------------:|:------:|
| `model` | `yolo26{n,s,m}.pt` | `yolo26s.yaml` | `yolo26s.pt` |
| `imgsz` | 640 | 640 | 640 |
| `batch` | 16 | 16 | 16 |
| `epochs` | 100 | 100 | 100 |
| `patience` | 50 | 50 | 50 |
| `seed` | 42 | 42 | 42 |
| `augmentation` | Default | Default | Disabled |
| `pretrained` | COCO | None | COCO |

Default augmentation includes: HSV shift, horizontal flip, translate, scale, mosaic,
copy-paste (all Ultralytics defaults for detect task).

---

## Evaluation

After training, evaluate on the test split:

```bash
yolo detect val \
    model=runs/detect/y26n_vanilla/weights/best.pt \
    data=SawitMVC-YOLO/data.yaml \
    split=test
```

Expected metrics for y26n_vanilla:

| Metric | Value |
|--------|------:|
| mAP50 | 0.521 |
| mAP50-95 | 0.237 |
| Precision | ~0.65 |
| Recall | ~0.61 |

---

## Ablation Insights

### Augmentation is critical

Without augmentation, the YOLO26s model overfits at epoch 6 and achieves only
mAP50=0.465 (−0.041 vs vanilla). The training loss continues to decrease but
validation mAP degrades immediately after epoch 6. Default Ultralytics augmentation
must be enabled for any serious training run.

### COCO pretraining is optional

YOLO26s trained from scratch achieves mAP50=0.511 at epoch 57, **exceeding** the
pretrained vanilla YOLO26s (mAP50=0.506 at epoch 21). For this specific domain
(aerial/close-up oil palm images), ImageNet/COCO weights provide no advantage.
Training from scratch requires ~3× more epochs to converge.

### Nano outperforms medium

YOLO26n (mAP50=0.521) outperforms YOLO26m (mAP50=0.509) while being 8× smaller and
4× faster. Larger architectures tend to overfit on this dataset of ~4,000 images.
The nano model's implicit regularization from reduced capacity is beneficial.

---

## What Has Been Tried and Did Not Help

Based on 13+ training experiments (archived in the research repository):

- ❌ `imgsz=800` — no improvement, 40% slower
- ❌ Focal loss — no improvement
- ❌ Naive class oversampling — destabilized training
- ❌ SGD / AdamW sweep — Ultralytics auto-optimizer matches or beats manual selection
- ❌ `label_smoothing=0.1` — no improvement
- ❌ YOLOv9e / RT-DETR / RF-DETR — did not beat YOLO26n within compute budget
- ❌ Two-stage classifier (DINOv2 + EfficientNet) — violated no-training constraint and
  did not transfer well to the B2↔B3 ambiguity

---

## Using Pre-trained Weights

If you don't want to retrain, use the weights in `models/`:

```python
from ultralytics import YOLO

model = YOLO("models/y26n_vanilla_local.pt")
results = model.predict("path/to/tree_side_1.jpg", conf=0.25, iou=0.45)
```

See `models/README.md` for full inference examples.
