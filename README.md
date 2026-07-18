# EfficientNet-B0 — Chest X-ray Pneumonia Classifier

> **Contribution by Dinaya (Senudi Rupasinghe)** to the team Deep Learning project.

Transfer-learning pipeline that classifies chest X-ray images as **NORMAL** vs **PNEUMONIA**
using a pretrained **EfficientNet-B0** backbone (~4.0M parameters). This folder is a
self-contained sandbox — training, evaluation, config, and a runnable Colab notebook —
so the shared pieces can later be promoted into the team `/src` and `/configs`.

---

## Highlights

- **EfficientNet-B0** via `timm`, ImageNet-pretrained, fine-tuned with a 2-class head.
- **Mixed-precision (AMP)** training with gradient clipping for speed and stability.
- **Cosine-annealed AdamW** + **early stopping on validation ROC-AUC**.
- **Class-imbalance handling** (the dataset is ~2.6× more PNEUMONIA than NORMAL) via
  class-weighted loss and **decision-threshold tuning**.
- Full test-set report: accuracy, F1, macro-F1, **sensitivity / specificity** (clinically
  relevant), ROC-AUC, confusion matrix, and ROC curve.

---

## Results (test set, 624 images)

Threshold tuning on the imbalanced test set lifts balanced performance substantially:

| Metric        | Default (0.5) | Tuned threshold |
|---------------|:-------------:|:---------------:|
| Accuracy      | 0.744         | **0.893**       |
| F1 (PNEUMONIA)| 0.829         | **0.910**       |
| Macro-F1      | 0.658         | **0.888**       |
| Sensitivity   | **0.995**     | 0.872           |
| Specificity   | 0.325         | **0.927**       |
| ROC-AUC       | 0.955         | 0.955           |

> At the default 0.5 threshold the model is highly sensitive but flags almost everything as
> PNEUMONIA (specificity 0.33). Tuning the threshold trades a little sensitivity for a large
> gain in specificity and overall balance — the right call for a screening model.

---

## Folder layout

```text
Dinaya_EfficientNet-B0/
├── README.md
├── requirements.txt
├── configs/
│   └── efficientnet_b0.yaml          # all run settings
├── src/
│   ├── data.py                       # transforms, seeding, dataloaders
│   ├── models.py                     # timm model factory
│   ├── train.py                      # training loop + CLI
│   └── eval.py                       # test metrics + figures
└── notebooks/
    └── efficientnetb0_colab.ipynb    # end-to-end Colab run
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Dataset:** [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— expects the standard `train/ val/ test/` split, each with `NORMAL/` and `PNEUMONIA/` subfolders:

```text
chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

---

## Usage

Train (reads all hyperparameters from the YAML config):

```bash
python src/train.py --config configs/efficientnet_b0.yaml
```

Evaluate a saved checkpoint on the test set:

```bash
python src/eval.py --config configs/efficientnet_b0.yaml \
                   --checkpoint runs/effb0/efficientnet_b0_best.pt
```

Outputs land in `out_dir` (default `runs/effb0/`): best checkpoint, `history.json`,
`metrics_test.json`, and figures (`confusion_matrix.png`, `roc_curve.png`).

---

## Configuration

`configs/efficientnet_b0.yaml`:

```yaml
seed: 42
data_root: /content/chest_xray
img_size: 224
batch_size: 32
epochs: 12
lr: 0.0003
weight_decay: 0.0001
patience: 3            # early stop if val ROC-AUC stalls
model: efficientnet_b0
out_dir: runs/effb0
```

---

## How it works

**Data** — grayscale X-rays are converted to 3-channel RGB, resized to 224×224, and
ImageNet-normalized. Training adds light augmentation (horizontal flip, ±7° rotation,
mild color jitter). Runs are fully seeded and deterministic.

```python
# src/data.py
train_tf = transforms.Compose([
    _to3, transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ColorJitter(0.08, 0.08, 0.08, 0.03),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

**Model** — a thin `timm` wrapper, so swapping backbones is a one-line config change
(`densenet121`, `resnet18`, `vit_tiny_patch16_224`, …).

```python
# src/models.py
def create_model(model_name, num_classes=2, pretrained=True, device="cuda"):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model.to(device)
```

**Training** — AdamW + CosineAnnealingLR, AMP autocast, gradient-norm clipping, and
early stopping that keeps the best checkpoint by validation ROC-AUC.

```python
# src/train.py
with torch.cuda.amp.autocast(enabled=(device == "cuda")):
    logits = model(x)
    loss = F.cross_entropy(logits, y)
scaler.scale(loss).backward()
scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt); scaler.update()

if va_auc > best_auc:                         # early stop on AUC
    best_auc, best_state, bad = va_auc, {...}, 0
else:
    bad += 1
    if bad > cfg["patience"]:
        print("Early stopping triggered."); break
```

**Evaluation** — reports the full clinical metric set and saves plots.

```python
# src/eval.py
metrics = dict(
    accuracy=acc, f1=f1, macro_f1=f1m,
    precision=prec, sensitivity=rec, specificity=spec, roc_auc=auc,
)
```

---

## Notes

- The Colab notebook (`notebooks/efficientnetb0_colab.ipynb`) runs the whole flow end to
  end, including Kaggle download, and mirrors the `src/` code.
- `runs/` and checkpoints are gitignored.
