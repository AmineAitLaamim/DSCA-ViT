# Ensemble -- shared val-first / test-once evaluation driver.
# Each variant script calls run_variant() with its own constants.
#   Default   : eval on val_indices, save val_results.*
#   --eval-test: eval full test_dir once (locked), save test_results.*
#   --debug   : run sanity checks.
# Ensemble = equal-weight average of per-model PROBS (never logits).

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

# --- sys.path setup (Ensemble/common importable; datasets importable) ---
ENSEMBLE_DIR = Path(__file__).resolve().parent.parent
if str(ENSEMBLE_DIR) not in sys.path:
    sys.path.insert(0, str(ENSEMBLE_DIR))
PROJECT_ROOT = ENSEMBLE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.model_loader import build_and_load_models
from common.metrics_utils import compute_metrics, save_results
from common.dataset_utils import build_eval_dataset, load_split
from common.inference import average_ensemble_probs, collect_probs, ensemble_predictions
from common.test_lock import check_and_acquire, write_lock


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble evaluation (validation-first)")
    p.add_argument("--config", type=str, default="configs/ensemble_config.yaml")
    p.add_argument("--eval-test", action="store_true",
                   help="Evaluate on the official test set (default: validation).")
    p.add_argument("--force", action="store_true",
                   help="Override the test-evaluation lock.")
    p.add_argument("--debug", action="store_true",
                   help="Run debug sanity checks instead of a full eval.")
    return p.parse_args()


def _infer(models_info, root_dir, indices, model_names, tta, bs, nw, device):
    """Collect per-model probs, average them -> (probs, labels, preds)."""
    probs_list, labels = [], None
    for name, model, dev in models_info:
        print(f"  [infer] running '{name}' (tta={tta}) ...")
        start = time.time()
        ds = build_eval_dataset(root_dir, indices, name, tta=tta)
        probs, lbl = collect_probs(model, dev, ds, name, tta=tta,
                                   batch_size=bs, num_workers=nw)
        probs_list.append(probs)
        labels = lbl if labels is None else labels
        print(f"    -> {name}: probs {probs.shape} in {time.time()-start:.1f}s")
    avg = average_ensemble_probs(probs_list)
    return avg, labels, ensemble_predictions(avg)


def run_variant(variant_name, experiment_name, model_names, tta) -> None:
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ensemble variant: {variant_name} (experiment {experiment_name})")
    print(f"Constituents: {model_names} | TTA: {tta} | device: {device}")

    # Nested two-level log/results paths.
    log_dir = os.path.join(config["paths"]["log_dir_base"], variant_name, experiment_name)
    results_dir = os.path.join(config["paths"]["results_dir_base"], variant_name, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"  log_dir    : {log_dir}")
    print(f"  results_dir: {results_dir}")

    bs = config["experiment"]["batch_size"]
    nw = config["experiment"]["num_workers"]

    if args.debug:
        from common.debug_checks import run_all_debug_checks
        run_all_debug_checks(model_names, config, device,
                             tempfile.mkdtemp(prefix="ensemble_debug_"))
        return

    models_info = build_and_load_models(model_names, config, device)
    print(f"Loaded {len(models_info)} model(s): {[n for n, _, _ in models_info]}")

    if not args.eval_test:
        # ---- VALIDATION (default) ----
        print("=" * 60)
        print("VALIDATION EVALUATION")
        print("=" * 60)
        val_indices, _test = load_split(config["paths"]["split_indices_path"])
        _probs, labels, preds = _infer(models_info, config["paths"]["train_dir"],
                                       val_indices, model_names, tta, bs, nw, device)
        metrics = compute_metrics(labels, preds)
        extra = {"split": "val", "variant_name": variant_name,
                 "experiment_name": experiment_name, "constituents": model_names,
                 "tta": tta, "num_samples": int(len(labels))}
        save_results(results_dir, "val", metrics, extra)
        return

    # ---- OFFICIAL TEST (--eval-test, locked) ----
    lock_base = config["paths"]["results_dir_base"]
    check_and_acquire(lock_base, variant_name, args.force)
    print("=" * 60)
    print("OFFICIAL TEST EVALUATION (test-once, locked)")
    print("=" * 60)

    _probs, labels, preds = _infer(models_info, config["paths"]["test_dir"],
                                   None, model_names, tta, bs, nw, device)
    metrics = compute_metrics(labels, preds)
    extra = {"split": "test", "variant_name": variant_name,
             "experiment_name": experiment_name, "constituents": model_names,
             "tta": tta, "num_samples": int(len(labels))}
    save_results(results_dir, "test", metrics, extra)

    write_lock(lock_base, variant_name, metrics["accuracy"])
    print(f"  [test] accuracy: {metrics['accuracy']*100:.2f}%")


if __name__ == "__main__":
    run_variant("demo", "demo_001", ["vit_ce", "uni_ce"], tta=False)
