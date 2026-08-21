#!/usr/bin/env python3
# ============================================================
# HER2-IHC-40x — Inference Script
# DenseNet121 (Stage 2 fine-tuned checkpoint)
# ============================================================
#
# Usage:
#   Single image:
#     python her2_inference.py --checkpoint stage2_weights.pth --image path/to/img.png
#
#   Folder of images (batch), results written to CSV:
#     python her2_inference.py --checkpoint stage2_weights.pth --dir path/to/folder --output results.csv
#
# Checkpoint must be the dict format saved in Stage 2 of the training
# notebook, i.e. contains "model_state_dict" (and optionally
# "best_val_accuracy" / "epoch" for logging).
# ============================================================

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ------------------------------------------------------------
# Constants — must match training notebook exactly
# ------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["0", "1+", "2+", "3+"]  # class_0, class_1+, class_2+, class_3+
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

inference_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Rebuild DenseNet121 (4-class head) and load trained weights."""

    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, len(CLASS_NAMES))

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both a full checkpoint dict and a raw state_dict file
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    if isinstance(checkpoint, dict) and "best_val_accuracy" in checkpoint:
        print(f"Loaded checkpoint | epoch {checkpoint.get('epoch', '?')} "
              f"| val acc {checkpoint['best_val_accuracy']:.2f}%")
    else:
        print("Loaded checkpoint (raw state_dict, no metadata).")

    return model


# ------------------------------------------------------------
# Single-image inference
# ------------------------------------------------------------

def predict_image(model: torch.nn.Module, image_path: Path, device: torch.device) -> dict:
    image = Image.open(image_path).convert("RGB")
    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probs).item())

    return {
        "file": image_path.name,
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


def print_result(result: dict) -> None:
    print("=" * 60)
    print(f"File       : {result['file']}")
    print(f"Prediction : HER2 {result['predicted_class']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print("-" * 60)
    print("Class probabilities")
    for cls, p in result["probabilities"].items():
        bar = "#" * int(p * 30)
        print(f"  {cls:>3} | {p * 100:6.2f}% {bar}")
    print("=" * 60)


# ------------------------------------------------------------
# Batch inference over a folder
# ------------------------------------------------------------

def predict_folder(model: torch.nn.Module, folder: Path, device: torch.device) -> list:
    image_paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {folder}")

    results = []
    for i, img_path in enumerate(image_paths, 1):
        result = predict_image(model, img_path, device)
        results.append(result)
        print(f"[{i}/{len(image_paths)}] {img_path.name:40s} -> HER2 {result['predicted_class']} "
              f"({result['confidence'] * 100:.1f}%)")

    return results


def save_results_csv(results: list, output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "predicted_class", "confidence"] + [f"prob_{c}" for c in CLASS_NAMES])
        for r in results:
            writer.writerow([
                r["file"],
                r["predicted_class"],
                f"{r['confidence']:.4f}",
                *[f"{r['probabilities'][c]:.4f}" for c in CLASS_NAMES],
            ])
    print(f"\nResults saved to: {output_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HER2-IHC DenseNet121 inference")
    parser.add_argument("--checkpoint", required=True, help="Path to stage2_checkpoint.pth")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--dir", help="Path to a folder of images (batch mode)")
    parser.add_argument("--output", default="her2_predictions.csv", help="CSV output path for batch mode")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Provide either --image or --dir")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    model = load_model(str(checkpoint_path), device)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        result = predict_image(model, image_path, device)
        print_result(result)

    else:
        folder = Path(args.dir)
        if not folder.exists():
            print(f"Folder not found: {folder}", file=sys.stderr)
            sys.exit(1)
        results = predict_folder(model, folder, device)
        save_results_csv(results, Path(args.output))


if __name__ == "__main__":
    main()