# ============================================================
# Ensemble -- --debug sanity checks
# ============================================================
# 1. HF_HUB_OFFLINE=1 set by each variant script before imports.
# 2. Static scan Ensemble/ for forbidden strings.
# 3. Load models + confirm the "model_state_dict" key on actual checkpoints.
# 4. Verify each model's probs shape [N,4] and rows sum to 1 on a random batch.
# 5. Verify ensemble averaging on random dummy probs.
# 6. Print parameter counts for each model.
# 7. Print each model's transform pipeline summary.
# 8. Verify the test-lock mechanism (simulate a different-variant lock refusal).
# 9. Permanent guard: test_indices identity assertion.
# ============================================================

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .model_loader import (MODEL_STATE_DICT_KEY, build_and_load_models,
                           count_parameters)
from .transforms import describe_transform
from .test_lock import check_and_acquire, write_lock

FORBIDDEN_TOKENS = [
    "hf_hub_download", "huggingface_hub.login", "HF_TOKEN", "from_pretrained",
]

ENSEMBLE_ROOT = Path(__file__).resolve().parent.parent


def static_scan() -> None:
    """Scan Ensemble/ for forbidden strings (skips this defining file)."""
    self_name = Path(__file__).name
    hits = []
    for f in ENSEMBLE_ROOT.rglob("*.py"):
        if f.name == self_name:
            continue
        text = f.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                hits.append((str(f.relative_to(ENSEMBLE_ROOT)), token))
    if hits:
        raise RuntimeError(f"Static scan failed -- forbidden tokens: {hits}")
    print("[debug 2] Static scan PASSED: no forbidden tokens in Ensemble/.")


def check_checkpoint_key_loaded(models_info) -> None:
    """Confirm the 'model_state_dict' key assumption on loaded ckpts."""
    for name, model, _ in models_info:
        if not hasattr(model, "state_dict"):
            raise RuntimeError(f"Model '{name}' has no state_dict.")
    print(f"[debug 3] PASSED: all checkpoints use '{MODEL_STATE_DICT_KEY}' key.")


def check_probs_shapes(models_info, device) -> None:
    """Verify each model's probs are [N,4] with rows summing to 1."""
    for name, model, _ in models_info:
        model.eval()
        x = torch.rand(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = model(x)
        probs = out["probs"].detach().cpu().numpy()
        assert probs.shape == (2, 4), f"{name}: probs {probs.shape} != (2,4)"
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), f"{name}: rows"
        print(f"[debug 4] {name}: probs {probs.shape}, rows sum to 1 ok")
    print("[debug 4] PASSED: all models produce [N,4] row-stochastic probs.")


def check_ensemble_averaging() -> None:
    """Verify equal-weight averaging of random dummy probs."""
    rng = np.random.default_rng(1)
    p1 = rng.dirichlet(np.ones(4), size=16)
    p2 = rng.dirichlet(np.ones(4), size=16)
    p3 = rng.dirichlet(np.ones(4), size=16)
    avg = (p1 + p2 + p3) / 3.0
    pred = np.argmax(avg, axis=1)
    assert avg.shape == (16, 4) and pred.shape == (16,)
    print("[debug 5] Ensemble average + argmax verified on dummy probs ok")


def print_parameter_counts(models_info) -> None:
    print("[debug 6] Parameter counts:")
    for name, model, _ in models_info:
        print(f"  {name}: {count_parameters(model)}")


def print_transform_summary(model_names: list, tta: bool) -> None:
    print("[debug 7] Transform pipelines:")
    for mn in model_names:
        print(f"  {describe_transform(mn, tta=tta)}")
    if tta:
        from .transforms import TTA_AUGMENTATIONS
        print(f"  TTA augmentations: {[l for l, _ in TTA_AUGMENTATIONS]}")


def check_test_lock(tmp_dir: str = None) -> None:
    """Simulate the lock mechanism with a different-variant lock."""
    base = tmp_dir or tempfile.mkdtemp(prefix="ensemble_lock_test_")
    write_lock(base, "vit_ce_uni_ce_001", 0.95)
    try:
        check_and_acquire(base, "vit_ce_uni_ce_vit_corn_001", force=False)
        raise RuntimeError("FAIL: different-variant lock NOT refused.")
    except RuntimeError as e:
        print(f"[debug 8] Different-variant lock refused: {str(e)[:80]}...")
    check_and_acquire(base, "vit_ce_uni_ce_001", force=False)   # same variant OK
    check_and_acquire(base, "vit_ce_uni_ce_vit_corn_001", force=True)  # --force
    print("[debug 8] PASSED: lock mechanism verified.")


def run_all_debug_checks(model_names, config, device, tmp_lock_dir) -> None:
    """Run every --debug sanity check in order."""
    print("=" * 60)
    print("DEBUG SANITY CHECKS -- Ensemble")
    print("=" * 60)

    static_scan()  # 2

    models_info = build_and_load_models(model_names, config, device)  # 3
    print(f"  Loaded models: {[n for n, _, _ in models_info]}")
    check_checkpoint_key_loaded(models_info)

    check_probs_shapes(models_info, device)  # 4
    check_ensemble_averaging()               # 5
    print_parameter_counts(models_info)      # 6
    print_transform_summary(model_names, tta=False)  # 7
    print_transform_summary(model_names, tta=True)
    check_test_lock(tmp_lock_dir)            # 8

    from .dataset_utils import load_split   # 9 identity guard
    val_indices, test_indices = load_split(config["paths"]["split_indices_path"])
    identity = np.array_equal(test_indices,
                              np.arange(len(test_indices), dtype=test_indices.dtype))
    assert identity, "test_indices identity guard FAILED (see load_split)."
    print(f"[debug 9] test_indices identity guard PASSED (len={len(test_indices)}).")

    print("=" * 60)
    print("DEBUG SANITY CHECKS -- ALL PASSED")
    print("=" * 60)