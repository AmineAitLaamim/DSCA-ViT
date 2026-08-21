# HER2-IHC Hybrid Ensemble

Paper-faithful implementation of the final proposed HER2-IHC scoring model:

- EVA-02-Large
- ViT-Base/P16
- ConvNeXt-V2-Nano
- End-to-end feature fusion
- MLP classifier for HER2 classes `0`, `1+`, `2+`, `3+`

The implementation is designed for Google Colab and stores experiment outputs in persistent directories, preferably Google Drive.

## Install

```bash
pip install -r requirements.txt
```

## Colab Setup From GitHub

In Colab, first mount Drive, then clone or update this repository:

```python
from google.colab import drive
drive.mount('/content/drive')

REPO_URL = "https://github.com/AmineAitLaamim/HER2-model-classification.git"
REPO_DIR = "/content/HER2-model-classification"

import os

if os.path.exists(REPO_DIR):
    %cd {REPO_DIR}
    !git pull
else:
    !git clone {REPO_URL} {REPO_DIR}
    %cd {REPO_DIR}

!pip install -r requirements.txt
```

Use Google Drive for persistent data and results:

```python
DATA_DIR = "/content/drive/MyDrive/HER2_Classification/datasets/HER2-IHC-40x"
OUTPUT_DIR = "/content/drive/MyDrive/HER2_Classification/experiments"
CONFIG = "configs/hybrid.yaml"

!mkdir -p /content/drive/MyDrive/HER2_Classification/datasets
!mkdir -p /content/drive/MyDrive/HER2_Classification/experiments
!mkdir -p /content/drive/MyDrive/HER2_Classification/figures
!mkdir -p /content/drive/MyDrive/HER2_Classification/notebooks
```

## Dataset

The code supports a directory dataset:

```text
data/
  train/
    0/
    1+/
    2+/
    3+/
  val/
    0/
    1+/
    2+/
    3+/
  test/
    0/
    1+/
    2+/
    3+/
```

If no validation split exists, `prepare_data.py` and `train_hybrid.py` create a stratified validation split from training images.

## Prepare Manifests

```bash
python scripts/prepare_data.py --config configs/hybrid.yaml --data-dir /path/to/data --out-dir data
```

## Train

```bash
python scripts/train_hybrid.py --config configs/hybrid.yaml --data-dir /path/to/data --output-dir /content/drive/MyDrive/HER2_Classification/experiments
```

The default training is end-to-end. All three backbones and the fusion MLP are updated simultaneously.

## Evaluate

```bash
python scripts/evaluate_hybrid.py --config experiments/exp001_hybrid_end_to_end/config.yaml --checkpoint experiments/exp001_hybrid_end_to_end/best.pt --data-dir /path/to/data --output-dir outputs/eval
```

## Inference

```bash
python scripts/infer.py --config experiments/exp001_hybrid_end_to_end/config.yaml --checkpoint experiments/exp001_hybrid_end_to_end/best.pt --input /path/to/image_or_folder --output outputs/inference.json
```

## Outputs

Each experiment saves:

```text
config.yaml
environment.json
history.csv
training.log
best.pt
latest.pt
metrics.json
classification_report.json
predictions.csv
confusion_matrix.png
roc.png
pr.png
```

## Paper-Faithful Defaults

- epochs: `100`
- image size: `224`
- batch size: `32`
- optimizer: AdamW
- learning rate: `1e-5`
- weight decay: `0.01`
- warmup epochs: `10`
- scheduler: linear warmup + cosine
- MixUp alpha: `0.5`
- label smoothing: `0.1`
- EMA decay: `0.9999`
- gradient clip norm: `1.0`
- early stopping patience: `25`

Known paper ambiguities are documented in `IMPLEMENTATION_PLAN.md`.
