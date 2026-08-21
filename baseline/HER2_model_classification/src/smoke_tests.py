from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint, save_checkpoint
from .ema import ModelEMA
from .evaluate import evaluate_and_save
from .losses import MixupCriterion
from .models import build_model
from .train import build_scheduler


def run_smoke_tests(config, train_dataset, val_dataset, exp_dir: str | Path) -> None:
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Smoke tests require non-empty train and validation datasets.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(train_dataset, batch_size=min(2, len(train_dataset)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=min(2, len(val_dataset)), shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"]["amp"]) and device.type == "cuda")
    criterion = MixupCriterion(float(config["training"]["label_smoothing"]))
    ema = ModelEMA(model, float(config["training"]["ema_decay"]))

    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    with torch.cuda.amp.autocast(enabled=bool(config["training"]["amp"]) and device.type == "cuda"):
        logits = model(images)
        loss = criterion(logits, labels)
    if logits.shape != (images.shape[0], len(config["classes"])):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(loss):
        raise AssertionError("Smoke-test loss is not finite.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    ema.update(model)

    exp_dir = Path(exp_dir)
    tmp_ckpt = exp_dir / "smoke_test.pt"
    save_checkpoint(tmp_ckpt, model, optimizer, scheduler, scaler, ema, 0, float(loss.detach().cpu()), config)
    load_checkpoint(tmp_ckpt, model, optimizer, scheduler, scaler, ema, device)
    class_names = [k for k, _ in sorted(config["classes"].items(), key=lambda item: item[1])]
    evaluate_and_save(ema.ema, val_loader, device, class_names, exp_dir / "smoke_eval")

