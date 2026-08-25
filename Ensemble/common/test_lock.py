# ============================================================
# Ensemble — test-evaluation lock mechanism
# ============================================================
#
# The official test set is evaluated exactly once per variant.
# A shared lock file (results/Ensemble/.test_eval_lock.json)
# records which variant already ran on the official test set. If
# the lock exists and names a DIFFERENT variant than the one
# currently running, we refuse unless --force is passed. If it
# does not exist, or already names the current variant, we
# proceed and write/update the lock.
# ============================================================

from __future__ import annotations

import json
import os
from typing import Optional

LOCK_FILENAME = ".test_eval_lock.json"


def _lock_path(base_dir: str) -> str:
    return os.path.join(base_dir, LOCK_FILENAME)


def read_lock(base_dir: str) -> Optional[dict]:
    """Return the lock dict, or None if it does not exist."""
    path = _lock_path(base_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return None


def check_and_acquire(base_dir: str, variant_name: str, force: bool) -> None:
    """Enforce the test-once lock. Raises RuntimeError on refusal."""
    lock = read_lock(base_dir)
    if lock is None:
        return  # no lock yet -> allowed
    locked_variant = lock.get("variant_evaluated_on_test")
    if locked_variant == variant_name:
        # Running the same variant again -> allowed (re-eval of same variant).
        return
    if not force:
        raise RuntimeError(
            f"Test-evaluation lock refusal: "
            f"'{_lock_path(base_dir)}' already names variant "
            f"'{locked_variant}', but the current run is '{variant_name}'. "
            "The official test set should be evaluated once per variant. "
            "Re-run with --force to override."
        )
    # force -> allowed, will overwrite below.


def write_lock(base_dir: str, variant_name: str, test_accuracy: float) -> None:
    """Write/update the lock file for the given variant."""
    os.makedirs(base_dir, exist_ok=True)
    import datetime
    payload = {
        "variant_evaluated_on_test": variant_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "test_accuracy": float(test_accuracy),
    }
    path = _lock_path(base_dir)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [lock] updated '{path}' -> {payload}")
