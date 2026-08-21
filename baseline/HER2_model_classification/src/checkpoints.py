from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "ema": ema.state_dict() if ema else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        out,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, ema=None, device="cpu") -> dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None and checkpoint.get("ema"):
        ema.load_state_dict(checkpoint["ema"])
    return checkpoint

