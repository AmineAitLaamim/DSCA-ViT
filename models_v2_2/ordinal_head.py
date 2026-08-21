# ============================================================
# DSS-ViT v2.2 — Ordinal Head & Loss Functions
# ============================================================
#
# Ordinal classification head for HER2 scoring {0, 1+, 2+, 3+}.
# The head predicts 3 cutpoints; class probabilities are derived
# from the cumulative distribution.
#
# Identical to models_v2_1/ordinal_head.py.
# ============================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OrdinalHead(nn.Module):
    """
    Ordinal classification head.

    Maps the fused CLS representation to `num_classes - 1` cutpoint
    logits. Class probabilities are derived from the sigmoid of the
    cutpoints (cumulative distribution).

    Args:
        in_dim (int): Input feature dimension (default 768).
        num_classes (int): Number of ordinal classes (default 4).
    """

    def __init__(self, in_dim: int = 768, num_classes: int = 4) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes - 1)  # 3 cutpoints

        # Xavier init
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Fused CLS features, shape (B, in_dim).

        Returns:
            torch.Tensor: Raw cutpoint logits, shape (B, num_classes - 1).
        """
        return self.fc(x)


def cutpoints_to_probs(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Converts cutpoint logits to class probabilities.

    Args:
        logits (torch.Tensor): Raw cutpoint logits, shape (B, 3).
        eps (float): Small clamp value to avoid log(0).

    Returns:
        torch.Tensor: Class probabilities, shape (B, 4), clamped to [eps, 1-eps].
    """
    cutpoints = torch.sigmoid(logits)  # (B, 3)

    p0 = 1.0 - cutpoints[:, 0]
    p1 = cutpoints[:, 0] - cutpoints[:, 1]
    p2 = cutpoints[:, 1] - cutpoints[:, 2]
    p3 = cutpoints[:, 2]

    probs = torch.stack([p0, p1, p2, p3], dim=1)  # (B, 4)
    probs = torch.clamp(probs, min=eps, max=1.0 - eps)

    return probs


def ordinal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Ordinal loss via binary cross-entropy on the cumulative cutpoints.

    Args:
        logits (torch.Tensor): Raw cutpoint logits, shape (B, num_classes - 1).
        targets (torch.Tensor): Integer labels, shape (B,).
        num_classes (int): Number of ordinal classes.

    Returns:
        torch.Tensor: Scalar ordinal loss.
    """
    # Ordinal targets: [B, num_classes-1]
    ord_targets = (
        targets.unsqueeze(1) > torch.arange(num_classes - 1, device=targets.device)
    ).float()

    return F.binary_cross_entropy_with_logits(logits, ord_targets, reduction="mean")


def total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    probs: torch.Tensor,
    alpha: float = 0.1,
    label_smoothing: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined cross-entropy + ordinal loss.

    Args:
        logits (torch.Tensor): Raw cutpoint logits, shape (B, 3).
        targets (torch.Tensor): Integer labels, shape (B,).
        probs (torch.Tensor): Class probabilities, shape (B, 4).
        alpha (float): Weight of the ordinal loss term.
        label_smoothing (float): Label smoothing for cross-entropy.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (total_loss, ce_loss, ordinal_loss).
    """
    ce = F.cross_entropy(probs.log(), targets, label_smoothing=label_smoothing)
    ord = ordinal_loss(logits, targets)
    total = ce + alpha * ord
    return total, ce, ord