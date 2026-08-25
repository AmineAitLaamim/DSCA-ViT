# ============================================================
# Ensemble — shared model loader
# ============================================================
#
# Loads already-trained checkpoints (read-only inputs) for the
# three constituent models and exposes them as a list of
# (name, eval-mode model, device).
#
# IMPORTANT (checkpoint key):
#   Every checkpoint written by this project's train scripts uses
#   the "model_state_dict" key (see baseline/train_baseline_vit.py,
#   ViT-B16-CORN/train_vit_b16_corn.py). We hardcode that key and
#   assert it at load time so a future format change fails loudly.
#
# IMPORTANT (pretrained=False):
#   We load weights ONLY from best_stage2.pt via load_state_dict.
#   pretrained=False avoids any ImageNet weight download from
#   Hugging Face / timm — which would fail under HF_HUB_OFFLINE=1.
# ============================================================

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# ------------------------------------------------------------
# sys.path setup so the model classes can be imported by path
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HYPHEN_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "baseline" / "UNI-baseline",
    PROJECT_ROOT / "ViT-B16-CORN",
]
for _p in _HYPHEN_DIRS:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from baseline.models_baseline import PlainViTB16          # noqa: E402
from uni_baseline_model import UNIBaselineModel            # noqa: E402
from vit_b16_corn_model import ViTB16CORN                  # noqa: E402

# Checkpoint key convention — confirmed from save_checkpoint() source.
MODEL_STATE_DICT_KEY = "model_state_dict"


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return parameter counts compatible with each model's helper."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def load_strict(model: nn.Module, checkpoint_path: str, device) -> None:
    """Load best_stage2.pt and assert the 'model_state_dict' key."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. This ensemble "
            "evaluates already-trained models; the checkpoint must exist."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if MODEL_STATE_DICT_KEY not in ckpt:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' does not contain "
            f"'{MODEL_STATE_DICT_KEY}'. Found keys: {list(ckpt.keys())}. "
            "The ensemble loader assumes the project-standard "
            "'model_state_dict' key."
        )
    state = ckpt[MODEL_STATE_DICT_KEY]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"load_state_dict mismatch for '{checkpoint_path}':\n"
            f"  missing    : {missing[:20]}\n"
            f"  unexpected : {unexpected[:20]}"
        )
    model.eval()


def build_model(name: str, config: dict, device) -> nn.Module:
    """Construct a constituent model for the given name (no pretrained)."""
    if name == "vit_ce":
        model = PlainViTB16(num_classes=4, pretrained=False)
    elif name == "uni_ce":
        uni_backbone = config["checkpoints"]["uni_backbone"]
        if not os.path.exists(uni_backbone):
            raise FileNotFoundError(
                f"UNI backbone checkpoint not found at '{uni_backbone}'. "
                "UNIBaselineModel must be constructed from the local UNI "
                "weights before its fine-tuned state_dict is applied."
            )
        model = UNIBaselineModel(checkpoint_path=uni_backbone, num_classes=4)
    elif name == "vit_corn":
        model = ViTB16CORN(num_classes=4, pretrained=False)
    else:
        raise ValueError(f"Unknown model name '{name}'")
    return model.to(device)


def build_and_load_models(
    model_names: List[str], config: dict, device
) -> List[Tuple[str, nn.Module, torch.device]]:
    """Build + load each checkpoint. Returns [(name, model, device), ...]."""
    checkpoint_paths = config["checkpoints"]
    loaded: List[Tuple[str, nn.Module, torch.device]] = []
    for name in model_names:
        ckpt_key = {"vit_ce": "vit_ce", "uni_ce": "uni_ce",
                    "vit_corn": "vit_corn"}[name]
        ckpt_path = checkpoint_paths[ckpt_key]
        model = build_model(name, config, device)
        load_strict(model, ckpt_path, device)
        loaded.append((name, model, device))
    return loaded
