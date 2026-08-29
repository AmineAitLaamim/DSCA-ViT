# ============================================================
# UNI-Stain-Attention — Global Stain Statistics Loader/Saver
# ============================================================
#
# Loads/saves the global H/DAB mean & std statistics computed on
# the TRAINING split only. These are used to normalize the stain
# channels inside the model forward pass.
#
# Self-contained copy of UNI_v2/stain_stats.py (identical
# implementation). Existing packages are NOT modified.
# ============================================================

from __future__ import annotations

import json
import os
from typing import Dict


def save_stain_stats(
    h_mean: float,
    h_std: float,
    dab_mean: float,
    dab_std: float,
    path: str,
) -> None:
    """
    Saves the global stain statistics to a JSON file.

    Args:
        h_mean (float): Mean of the Hematoxylin channel.
        h_std (float): Std of the Hematoxylin channel.
        dab_mean (float): Mean of the DAB channel.
        dab_std (float): Std of the DAB channel.
        path (str): Output JSON path.
    """
    stats = {
        "h_mean": float(h_mean),
        "h_std": float(h_std),
        "dab_mean": float(dab_mean),
        "dab_std": float(dab_std),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stain statistics saved to '{path}'")


def load_stain_stats(path: str) -> Dict[str, float]:
    """
    Loads the global stain statistics from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Dict[str, float]: Dictionary with keys
            h_mean, h_std, dab_mean, dab_std.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stain statistics file not found at '{path}'. "
            "Run UNI-Stain-Attention/precompute_stain_stats.py first."
        )

    with open(path, "r") as f:
        stats = json.load(f)

    required_keys = ["h_mean", "h_std", "dab_mean", "dab_std"]
    for key in required_keys:
        if key not in stats:
            raise ValueError(
                f"Stain statistics file '{path}' is missing key '{key}'. "
                f"Expected keys: {required_keys}"
            )

    return stats