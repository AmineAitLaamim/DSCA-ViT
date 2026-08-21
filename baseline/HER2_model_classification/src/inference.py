from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .checkpoints import load_checkpoint
from .models import build_model
from .transforms import build_transforms


def load_inference_model(config: dict[str, Any], checkpoint_path: str | Path, device: str | torch.device | None = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    if checkpoint.get("ema"):
        model.load_state_dict(checkpoint["ema"])
    model.eval()
    return model, device


@torch.no_grad()
def predict_image(model, image_path: str | Path, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    transform = build_transforms(config, "test")
    class_names = [k for k, _ in sorted(config["classes"].items(), key=lambda item: item[1])]
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    return {
        "image_path": str(image_path),
        "predicted_label": class_names[pred_idx],
        "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
    }


def predict_path(model, path: str | Path, config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    path = Path(path)
    extensions = set(config["dataset"]["allowed_extensions"])
    if path.is_file():
        return [predict_image(model, path, config, device)]
    images = [p for p in path.rglob("*") if p.suffix.lower() in extensions]
    return [predict_image(model, p, config, device) for p in sorted(images)]

