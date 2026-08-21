from __future__ import annotations

import torch
from torch import nn


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


class MixupCriterion(nn.Module):
    def __init__(self, label_smoothing: float):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, y_a, y_b=None, lam: float = 1.0):
        if y_b is None:
            return self.ce(logits, y_a)
        return lam * self.ce(logits, y_a) + (1.0 - lam) * self.ce(logits, y_b)

