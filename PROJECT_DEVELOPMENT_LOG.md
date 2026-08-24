# DSCA-ViT Project — Development Log & Achievement Summary

## 1. Overview
- **Task**: HER2-IHC-40x breast cancer grading (4 ordinal classes: class_0, class_1+, class_2+, class_3+).
- **Data**: WSI patches under `/home/amine.aitlaamim-ext/projects/DSCA-ViT/data/HER2/`.
- **Shared split**: `split_indices_wsi.npz` (train 7,283 | val 810 | test 1,847; val_fraction=0.10, seed=42), WSI-aware, load-only.
- **Eval**: single official test; full metric suite (acc, bal-acc, macro/weighted F1, per-class P/R/F1, QWK, confusion matrix) + MAE where applicable.

## 2. Reference results
| Model | Test Acc | Bal Acc | Macro F1 |
|-------|----------|---------|----------|
| Proper ViT-B16 baseline | 94.69% | 93.58% | 91.11% |
| UNI original fine-tune | 94.69% | 92.77% | 91.35% |
| ViT-B16 CORN | 94.26% | 93.52% | 92.03% |
| DSS-ViT v2.2 | 93.18% | - | - |
| UNI-Stain-MLP | 92.91% | - | - |
| UNI-RGB-MLP | 93.23% | - | - |
| UNI-Regularized | 93.23% | 90.96% | 87.43% |
| UNI-Stain-Attention | 93.77% | 90.60% | 88.87% |
| UNI-CORN | TBD | TBD | TBD |

## 3. Existing packages (inspected, not modified)
- `baseline/` (proper ViT-B16), `models_v2_2/` (DSS-ViT v2.2), `utils/`, `models/`, `models_v2/`, `models_v2_1/`, `models_v3/`.

## 4. Newly created packages (chronological)
All self-contained (model+train+eval+README+config+SLURM); hyphen folders run by path; `HF_HUB_OFFLINE=1`; load-only split; `uv run python`; checkpoint key `"model_state_dict"`.

### 4.1 `baseline/UNI-baseline/` — UNI baseline
- UNI recipe: `timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)`, strict local load, internal ImageNet norm.
- 2-stage: frozen+head Adam 1e-4 (30ep) / full fine-tune 1e-5+1e-4 (30ep). Checkpoints best_stage1/2, last.

### 4.2 `UNI_v2/` — UNI-Stain-MLP
- Frozen UNI (train() override keeps eval), ColorDeconvolution (Ruifrok H-DAB), StainEncoder [B,2,H,W]->[B,512], self-consistent stain stats + precompute.
- Fusion cat([rgb 2048, stain 512]) -> MLP. Single-stage AdamW 1e-3, wd 0.05, batch 64, AMP, clip 1.0, LS 0.1.

### 4.3 `UNI-RGB-MLP/` — RGB-only control
- Same frozen-UNI, no stain branch, explicit torch.no_grad() around UNI forward. head Linear(2048,1024)-GELU-Dropout(0.3)-Linear(1024,4).

### 4.4 `UNI-Stain-Attention/` — stain-conditioned attention pooling
- Stain feature queries UNI patch tokens via nn.MultiheadAttention(256, 4 heads) -> stain_attended [B,256]. Fusion cat([2048,512,256]) -> MLP. --debug asserts [B,256].

### 4.5 `ViT-B16-CORN/` — ordinal regression baseline
- ViT-B16 (num_classes=0) + CORN head Linear(768,3). Official coral-pytorch: corn_loss, corn_label_from_logits. probs [B,4] from sigmoids (sums 1). argmax banned. ImageNet norm in dataloader. Metrics+MAE.

### 4.6 `UNI-Regularized/` — regularized fine-tuning
- UNI + AdamW (wd 0.05 non-1D via split_decay_params) + early stopping on val loss (patience 7, min_delta 1e-4), counters reset at stage transition; checkpoint by val acc.
- Pre-flight fix: original UNI-baseline never reloads best_stage1.pt at Stage 2; this loads it and logs "Stage 2 initialized from best_stage1.pt". Used actual key "model_state_dict".

### 4.7 `UNI-CORN/` — UNI + CORN (final)
- UNI backbone + CORN head Linear(1024,3), internal norm (raw RGB). 2-stage exactly as original UNI-baseline (Adam, no wd, no early stop). corn_loss + corn_label_from_logits; argmax banned. Stage 2 from best_stage1.pt. Metrics+MAE. README documents architecture + SLURM.

## 5. Key conventions
1. UNI recipe identical everywhere.
2. Checkpoint dict: model_state_dict, optimizer_state_dict, scheduler_state_dict, epoch, stage, metrics, config, split_indices_path.
3. Offline: HF_HUB_OFFLINE=1; --debug scans hf_hub_download, huggingface_hub.login, HF_TOKEN, from_pretrained.
4. CORN packages ban argmax.
5. Split: np.load only; raise if val_fraction != 0.10.
6. SLURM: --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=8, uv run python, absolute log paths.
7. Eval always loads best_stage2.pt/best.pt (never last.pt).

## 6. HPC (Toubkal) issues fixed
1. Stray XML fragments (`</`, `</write_to_file>`) leaked into configs/SLURM -> YAML ScannerError + shell syntax error. Fixed by rewrite; repo scan confirmed LEAKS: NONE.
2. Seed KeyError (config["training"]["seed"] -> config["experiment"]["seed"]) in ViT-B16-CORN.
3. cp1252 corruption (em-dash -> 0x97) from programmatic filter; fixed with clean UTF-8 + ASCII hyphens.
- Lesson: clean XML writes + repo-wide `</` scan + YAML parse before completion.

## 7. Achievements
- Created 7 self-contained experiment packages.
- Reproduced proper-ViT-B16 protocol in each.
- Integrated official coral-pytorch CORN (no hand-rolled ordinal).
- Diagnosed/fixed UNI-baseline Stage-2 init defect.
- Enforced offline + argmax-free prediction paths.
- Provided reference results table.
- Kept HPC-readiness consistent.