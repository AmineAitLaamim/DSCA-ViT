from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoints import load_checkpoint, save_checkpoint
from .ema import ModelEMA
from .evaluate import evaluate_and_save, predict
from .losses import MixupCriterion, mixup_batch
from .metrics import compute_metrics
from .models import build_model


def format_duration(seconds: float) -> str:
    """Render a duration as e.g. '1h02m03s', '2m03s', or '45s'."""
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def build_scheduler(optimizer, config: dict[str, Any]):
    epochs = int(config["training"]["epochs"])
    warmup = int(config["training"]["warmup_epochs"])
    min_lr = float(config["training"]["min_learning_rate"])
    max_lr = float(config["training"]["learning_rate"])
    min_factor = min_lr / max_lr

    def lr_lambda(epoch: int):
        if epoch < warmup:
            return min_factor + (1.0 - min_factor) * float(epoch + 1) / float(max(1, warmup))
        progress = float(epoch - warmup) / float(max(1, epochs - warmup))
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def append_history(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    exists = out.exists()
    with out.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    ema,
    device,
    config,
    epoch: int = 0,
    total_epochs: int = 1,
):
    print("[DEBUG] Entered train_one_epoch")
    model.train()
    total_loss = 0.0
    total_items = 0
    mixup_alpha = float(config["training"]["mixup_alpha"])
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"
    grad_clip = float(config["training"]["gradient_clip_norm"])

    print("[DEBUG] Building tqdm progress bar for loader...")
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{total_epochs} [train]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    print("[DEBUG] Starting batch loop...")
    for batch_idx, batch in enumerate(progress):
        print("[DEBUG] First batch loaded from loader")
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if batch_idx == 0:
            print(f"[DEBUG] images.device = {images.device}")
            print(f"[DEBUG] labels.device = {labels.device}")
        images, y_a, y_b, lam = mixup_batch(images, labels, mixup_alpha)
        print("[DEBUG] Running first forward pass...")

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, y_a, y_b, lam)
        print(f"[DEBUG] Forward pass done. loss = {loss.item():.6f}")
        scaler.scale(loss).backward()
        print("[DEBUG] Backward pass done")
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema:
            ema.update(model)
        print("[DEBUG] Optimizer step done")

        batch_size = images.size(0)
        batch_loss = float(loss.detach().cpu())
        total_loss += batch_loss * batch_size
        total_items += batch_size

        progress.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg=f"{total_loss / max(1, total_items):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    progress.close()
    scheduler.step()
    print("[DEBUG] Exiting train_one_epoch")
    return total_loss / max(1, total_items)


@torch.no_grad()
def validate_loss(
    model,
    loader,
    criterion,
    device,
    config,
    epoch: int = 0,
    total_epochs: int = 1,
):
    model.eval()
    total_loss = 0.0
    total_items = 0
    use_amp = bool(config["training"]["amp"]) and device.type == "cuda"

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{total_epochs} [val]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += float(loss.detach().cpu()) * images.size(0)
        total_items += images.size(0)
        progress.set_postfix(val_loss=f"{total_loss / max(1, total_items):.4f}")

    progress.close()
    return total_loss / max(1, total_items)


def train_hybrid(config, train_dataset, val_dataset, test_dataset, exp_dir: str | Path, logger):
    exp_dir = Path(exp_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Device detected: {device}")
    class_names = [k for k, _ in sorted(config["classes"].items(), key=lambda item: item[1])]

    print("[DEBUG] Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
    )
    print("[DEBUG] DataLoaders ready.")

    print("[DEBUG] Building model...")
    model = build_model(config).to(device)
    print("[DEBUG] Model built.")
    print(f"[DEBUG] Device: {device}")
    print(f"[DEBUG] Model device: {next(model.parameters()).device}")
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"]["amp"]) and device.type == "cuda")
    criterion = MixupCriterion(float(config["training"]["label_smoothing"]))
    print("[DEBUG] Creating EMA...")
    ema = ModelEMA(model, float(config["training"]["ema_decay"]))
    print("[DEBUG] EMA created.")

    start_epoch = 0
    best_metric = float("inf")
    latest = exp_dir / "latest.pt"
    if bool(config["training"].get("resume", True)) and latest.exists():
        print("[DEBUG] latest.pt exists, loading checkpoint...")
        ckpt = load_checkpoint(latest, model, optimizer, scheduler, scaler, ema, device)
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt["best_metric"])
        logger.info("Resumed from epoch %s", start_epoch)
        tqdm.write(f"Resumed from epoch {start_epoch} (best_metric={best_metric:.6f})")
    else:
        print("[DEBUG] No latest.pt found, starting fresh training.")

    total_epochs = int(config["training"]["epochs"])
    print(f"[DEBUG] start_epoch={start_epoch}")
    print(f"[DEBUG] total_epochs={total_epochs}")
    patience = int(config["training"]["early_stopping_patience"])
    stale_epochs = 0
    start_time = time.time()
    epoch_durations: list[float] = []
    logger.info("Starting training on device=%s", device)
    tqdm.write(f"Starting training on device={device} | epochs={total_epochs} | patience={patience}")

    if start_epoch >= total_epochs:
        print("[DEBUG] start_epoch >= total_epochs, skipping training loop!")
        tqdm.write(
            f"\u26a0 Nothing to train: resumed checkpoint is at epoch {start_epoch - 1}, "
            f"which already meets the configured {total_epochs} epochs. "
            f"Skipping straight to final evaluation on best.pt. "
            f"Increase 'training.epochs' in the config, or delete/rename latest.pt "
            f"to force a fresh run, if you intended to keep training."
        )
        logger.warning(
            "Resume epoch %s >= configured epochs %s; skipping training loop entirely, "
            "going straight to final evaluation.",
            start_epoch, total_epochs,
        )

    print("[DEBUG] Starting epoch loop...")
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        tqdm.write("")
        tqdm.write("=" * 60)
        tqdm.write(f"Epoch {epoch + 1}/{total_epochs}")
        tqdm.write("=" * 60)

        print("[DEBUG] Entering train_one_epoch...")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, ema, device, config,
            epoch=epoch, total_epochs=total_epochs,
        )
        print(f"[DEBUG] train_one_epoch returned. train_loss={train_loss:.6f}")
        eval_model = ema.ema if ema else model
        val_loss = validate_loss(
            eval_model, val_loader, criterion, device, config,
            epoch=epoch, total_epochs=total_epochs,
        )
        y_true, y_pred, y_prob, _ = predict(eval_model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        avg_epoch_duration = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = total_epochs - (epoch + 1)
        eta_seconds = avg_epoch_duration * remaining_epochs
        elapsed_seconds = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr": current_lr,
        }
        append_history(exp_dir / "history.csv", row)
        logger.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_bal_acc=%.6f lr=%.6e epoch_time=%.2fs elapsed=%.2fs eta=%.2fs",
            epoch, train_loss, val_loss, val_metrics["balanced_accuracy"], current_lr,
            epoch_duration, elapsed_seconds, eta_seconds,
        )

        tqdm.write(
            f"  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_bal_acc={val_metrics['balanced_accuracy']:.4f}  "
            f"lr={current_lr:.2e}"
        )
        tqdm.write(
            f"  epoch_time={format_duration(epoch_duration)}  "
            f"elapsed={format_duration(elapsed_seconds)}  "
            f"eta={format_duration(eta_seconds)}"
        )

        improved = val_loss < best_metric
        if improved:
            prev_best = best_metric
            best_metric = val_loss
            stale_epochs = 0
            save_checkpoint(exp_dir / "best.pt", model, optimizer, scheduler, scaler, ema, epoch, best_metric, config)
            tqdm.write(f"  \u2714 Saved best.pt  (val_loss improved {prev_best:.6f} -> {best_metric:.6f})")
            logger.info("Saved best.pt (val_loss improved from %.6f to %.6f)", prev_best, best_metric)
        else:
            stale_epochs += 1
            tqdm.write(f"  val_loss did not improve  ({stale_epochs}/{patience} epochs without improvement)")

        save_checkpoint(latest, model, optimizer, scheduler, scaler, ema, epoch, best_metric, config)
        tqdm.write("  \u2714 Saved latest.pt")
        logger.info("Saved latest.pt (epoch=%s)", epoch)

        if stale_epochs >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            logger.info("Early stopping at epoch %s", epoch)
            break

    total_duration = time.time() - start_time
    tqdm.write("")
    tqdm.write(f"Training complete in {format_duration(total_duration)}")
    logger.info("Training duration seconds=%.2f", total_duration)

    if (exp_dir / "best.pt").exists():
        load_checkpoint(exp_dir / "best.pt", model, ema=ema, device=device)
    eval_model = ema.ema if ema else model
    return evaluate_and_save(eval_model, test_loader, device, class_names, exp_dir)