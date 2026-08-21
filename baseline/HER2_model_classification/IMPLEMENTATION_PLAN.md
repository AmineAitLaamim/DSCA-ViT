# HER2-IHC Hybrid Ensemble Implementation Plan

## Objective

Implement the final proposed model from the paper as faithfully as possible: an end-to-end trainable hybrid ensemble for HER2-IHC scoring using EVA-02-Large, ViT-Base/P16, and ConvNeXt-V2-Nano with late feature concatenation and an MLP classifier.

The implementation prioritizes methodological fidelity over optimization. It intentionally excludes baseline-only training, cached-feature training, frozen-backbone training, XAI, robustness analysis, SHAP, ablations, and architectural improvements.

## Target Workflow

The workflow is designed for Google Colab with a T4 GPU.

1. Mount Google Drive.
2. Clone or pull the GitHub repository: `https://github.com/AmineAitLaamim/HER2-model-classification`.
3. Change into the cloned repository before installing dependencies.
4. Install dependencies from `requirements.txt`.
5. Prepare dataset manifests.
6. Run smoke tests.
7. Train the end-to-end hybrid ensemble.
8. Resume automatically from `latest.pt` if interrupted.
9. Evaluate the best checkpoint.
10. Save metrics, figures, predictions, checkpoints, logs, and configs to an experiment directory in Google Drive.

## Project Structure

```text
HER2_model_classification/
  configs/
    default.yaml
    hybrid.yaml

  src/
    __init__.py
    config.py
    dataset.py
    transforms.py
    models.py
    losses.py
    ema.py
    train.py
    evaluate.py
    inference.py
    checkpoints.py
    logging_utils.py
    metrics.py
    reproducibility.py
    smoke_tests.py
    utils.py

  scripts/
    prepare_data.py
    train_hybrid.py
    evaluate_hybrid.py
    infer.py

  notebooks/
    HER2_IHC_hybrid_colab.ipynb

  experiments/
    .gitkeep

  outputs/
    .gitkeep

  checkpoints/
    .gitkeep

  requirements.txt
  README.md
```

## Dataset Requirements

Support two input formats:

1. Directory format:

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

2. CSV manifest format:

```text
image_path,label,split,patient_id,wsi_id
```

Required labels:

```text
0  -> 0
1+ -> 1
2+ -> 2
3+ -> 3
```

Before training, verify class distribution, duplicate images, patient/WSI leakage when identifiers are available, and image readability. Abort training if train/validation/test leakage is detected.

## Paper-Faithful Defaults

Use YAML configs; do not hardcode experiment hyperparameters in source code.

Default choices:

```text
image_size: 224
epochs: 100
batch_size: 32
optimizer: AdamW
learning_rate: 1e-5
weight_decay: 0.01
warmup_epochs: 10
scheduler: linear warmup + cosine annealing
AMP: enabled
gradient_clip_norm: 1.0
EMA decay: 0.9999
early_stopping_patience: 25
MixUp alpha: 0.5
label_smoothing: 0.1
```

Documented assumptions:

- The paper reports 125 epochs in the scheduler equation but 100 epochs in the experiment section. The implementation defaults to 100 epochs.
- The paper reports MixUp alpha 0.2 in the preprocessing section and 0.5 in the experiment section. The implementation defaults to 0.5.
- The paper mentions 224 x 224 patches and also describes EVA at 448 input. The implementation defaults to 224 globally to match patch classification and ViT token count.
- The paper specifies label smoothing but not epsilon. The implementation defaults to 0.1.
- The paper specifies an MLP projection but not hidden size. The implementation defaults to 1024.

## Model Architecture

The hybrid model runs all backbones in every forward pass:

```text
Image
      |
 +----+----+
 |    |    |
 EVA  ViT  ConvNeXt
 |    |    |
 +----+----+
      |
Feature concatenation
      |
Fusion MLP
      |
4-class prediction
```

Backbone feature dimensions:

```text
EVA-02-Large      -> 1024
ViT-Base/P16      -> 768
ConvNeXt-V2-Nano  -> 640
Fusion vector     -> 2432
```

Fusion head:

```text
LayerNorm(2432)
Linear(2432, 1024)
GELU
Dropout(0.2)
Linear(1024, 4)
```

Gradients must update all trainable components simultaneously.

## Evaluation

Compute and save:

- Accuracy
- Balanced accuracy
- Weighted precision
- Weighted recall
- Weighted F1
- Per-class precision/recall/F1
- Specificity
- Confusion matrix
- ROC-AUC one-vs-rest
- PR curves
- `predictions.csv`

Saved files:

```text
metrics.json
classification_report.json
confusion_matrix.png
roc.png
pr.png
predictions.csv
```

## Experiment Tracking

Each run creates a new directory:

```text
experiments/exp001_hybrid/
```

Each experiment contains:

```text
config.yaml
history.csv
metrics.json
classification_report.json
training.log
best.pt
latest.pt
predictions.csv
confusion_matrix.png
roc.png
pr.png
```

Experiments must never overwrite previous runs.

## Checkpointing and Resume

Always save:

```text
best.pt
latest.pt
```

`latest.pt` contains model, optimizer, scheduler, AMP scaler, EMA state, epoch, best metric, and full config. Training resumes from `latest.pt` automatically when requested.

## Smoke Tests

Before long training, verify:

- Config loads.
- Dataset manifests are valid.
- Class mapping is correct.
- One batch loads.
- Transforms output tensors with expected shape.
- One forward pass returns logits shaped `batch_size x 4`.
- Loss computes.
- One optimizer step runs.
- EMA updates.
- Checkpoint save/load works.
- Evaluation on a tiny subset writes metrics.

## Final Deliverables

The final codebase will include configs, modular Python source, command-line scripts, a Colab notebook, README instructions, checkpoint handling, experiment tracking, evaluation figures, predictions export, and reproducibility metadata.
