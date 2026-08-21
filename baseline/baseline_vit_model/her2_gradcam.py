#!/usr/bin/env python3
# ============================================================
# HER2-IHC-40x — Grad-CAM Visualization
# DenseNet121 (Stage 2 fine-tuned checkpoint)
#
# Implements Grad-CAM (Selvaraju et al., 2020) the same way the
# reference paper applies it to DenseNet121:
#   - target layer = last conv block ("model.features", i.e. the
#     feature maps right before global average pooling)
#   - alpha_k^c = GAP over spatial dims of dL/dA_k  (Eq. 8)
#   - L_gradcam = ReLU( sum_k alpha_k^c * A_k )      (Eq. 7)
# ============================================================
#
# Usage:
#   Single image, heatmap for the PREDICTED class:
#     python her2_gradcam.py --checkpoint stage2_weights.pth --image sample.png
#
#   Single image, heatmap for a SPECIFIC class (0, 1+, 2+, 3+):
#     python her2_gradcam.py --checkpoint stage2_weights.pth --image sample.png --target-class 2+
#
#   Grid comparing all 4 class activations for one image (paper Fig. 8/9 style):
#     python her2_gradcam.py --checkpoint stage2_weights.pth --image sample.png --all-classes
#
#   Batch over a folder (predicted-class heatmap for each image):
#     python her2_gradcam.py --checkpoint stage2_weights.pth --dir test_images/ --output-dir cams/
# ============================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ------------------------------------------------------------
# Constants — must match training notebook exactly
# ------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["0", "1+", "2+", "3+"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ------------------------------------------------------------
# Model loading (same checkpoint format as her2_inference.py)
# ------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_features, len(CLASS_NAMES))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------
# Grad-CAM core
# ------------------------------------------------------------

class GradCAM:
    """
    Hooks the last conv block of DenseNet121 (model.features) and
    computes a class-discriminative localization map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Returns (cam [H,W] in [0,1], target_class used, confidence for that class).
        """
        self.model.zero_grad()

        logits = self.model(input_tensor)          # (1, 4)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]               # (C, H, W)
        activations = self.activations[0]            # (C, H, W)

        # alpha_k^c = GAP over spatial dims of the gradient (Eq. 8)
        weights = gradients.mean(dim=(1, 2))          # (C,)

        # L_gradcam = ReLU( sum_k alpha_k^c * A_k )   (Eq. 7)
        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = F.relu(cam)

        cam -= cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy(), target_class, float(probs[0, target_class].item())

    def remove(self):
        self._forward_handle.remove()


# ------------------------------------------------------------
# Overlay helpers
# ------------------------------------------------------------

def cam_to_overlay(cam: np.ndarray, original_image: Image.Image, alpha: float = 0.45) -> Image.Image:
    """Resize a [0,1] CAM to the original image size and blend with a jet colormap."""

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(original_image.size, resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0

    heatmap = cm.jet(cam_resized)[:, :, :3]  # drop alpha channel from colormap
    heatmap = np.uint8(heatmap * 255)

    base = np.asarray(original_image.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * base + alpha * heatmap.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def resolve_target_class(name: str) -> int:
    if name not in CLASS_NAMES:
        raise ValueError(f"--target-class must be one of {CLASS_NAMES}, got '{name}'")
    return CLASS_NAMES.index(name)


# ------------------------------------------------------------
# Single-image modes
# ------------------------------------------------------------

def run_single(gradcam: GradCAM, image_path: Path, device: torch.device,
                target_class: int, output_path: Path, alpha: float):

    original = Image.open(image_path).convert("RGB")
    tensor = preprocess(original).unsqueeze(0).to(device)

    cam, used_class, confidence = gradcam.generate(tensor, target_class=target_class)
    overlay = cam_to_overlay(cam, original.resize(IMAGE_SIZE), alpha=alpha)

    overlay.save(output_path)

    print(f"Image      : {image_path.name}")
    print(f"Class shown: HER2 {CLASS_NAMES[used_class]} (confidence {confidence * 100:.2f}%)")
    print(f"Saved to   : {output_path}")


def run_all_classes(gradcam: GradCAM, image_path: Path, device: torch.device,
                     output_path: Path, alpha: float):
    """Grid of original + Grad-CAM for each of the 4 HER2 classes, paper Fig. 8/9 style."""

    original = Image.open(image_path).convert("RGB")
    resized = original.resize(IMAGE_SIZE)
    tensor = preprocess(original).unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(resized)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for i, cls_name in enumerate(CLASS_NAMES):
        cam, _, confidence = gradcam.generate(tensor, target_class=i)
        overlay = cam_to_overlay(cam, resized, alpha=alpha)

        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"HER2 {cls_name}\n({confidence * 100:.1f}%)")
        axes[i + 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Image   : {image_path.name}")
    print(f"Saved to: {output_path}")


def run_batch(gradcam: GradCAM, folder: Path, device: torch.device,
              output_dir: Path, alpha: float):

    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise RuntimeError(f"No images found in {folder}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths, 1):
        out_path = output_dir / f"{img_path.stem}_gradcam.png"
        original = Image.open(img_path).convert("RGB")
        tensor = preprocess(original).unsqueeze(0).to(device)

        cam, used_class, confidence = gradcam.generate(tensor, target_class=None)
        overlay = cam_to_overlay(cam, original.resize(IMAGE_SIZE), alpha=alpha)
        overlay.save(out_path)

        print(f"[{i}/{len(image_paths)}] {img_path.name:40s} -> HER2 {CLASS_NAMES[used_class]} "
              f"({confidence * 100:.1f}%) -> {out_path.name}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for HER2-IHC DenseNet121")
    parser.add_argument("--checkpoint", required=True, help="Path to stage2_checkpoint.pth")
    parser.add_argument("--image", help="Path to a single image")
    parser.add_argument("--dir", help="Path to a folder of images (batch mode)")
    parser.add_argument("--target-class", choices=CLASS_NAMES, default=None,
                         help="Class to visualize (default: model's predicted class)")
    parser.add_argument("--all-classes", action="store_true",
                         help="With --image, produce a 1x5 grid (input + all 4 classes)")
    parser.add_argument("--output", default=None, help="Output path (single-image modes)")
    parser.add_argument("--output-dir", default="gradcam_outputs", help="Output folder (batch mode)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap blend strength (0-1)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.error("Provide either --image or --dir")
    if args.all_classes and not args.image:
        parser.error("--all-classes requires --image")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    model = load_model(str(checkpoint_path), device)
    gradcam = GradCAM(model, model.features)  # last conv block, matches the paper's target layer

    try:
        if args.image:
            image_path = Path(args.image)
            if not image_path.exists():
                print(f"Image not found: {image_path}", file=sys.stderr)
                sys.exit(1)

            if args.all_classes:
                output_path = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_gradcam_allclasses.png")
                run_all_classes(gradcam, image_path, device, output_path, args.alpha)
            else:
                target_class = resolve_target_class(args.target_class) if args.target_class else None
                output_path = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_gradcam.png")
                run_single(gradcam, image_path, device, target_class, output_path, args.alpha)

        else:
            folder = Path(args.dir)
            if not folder.exists():
                print(f"Folder not found: {folder}", file=sys.stderr)
                sys.exit(1)
            run_batch(gradcam, folder, device, Path(args.output_dir), args.alpha)

    finally:
        gradcam.remove()


if __name__ == "__main__":
    main()