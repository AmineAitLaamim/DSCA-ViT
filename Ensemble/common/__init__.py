# ============================================================
# Ensemble — common shared utilities
# ============================================================
#
# Self-contained helpers used by all four ensemble variants:
#   - model_loader.py   : load individual trained checkpoints
#   - transforms.py     : per-model transforms + TTA
#   - metrics_utils.py  : metrics computation (sklearn suite)
#   - dataset_utils.py  : split loading, dataset creation
#   - inference.py      : inference + probability ensembling
#   - test_lock.py      : test-evaluation lock mechanism
#   - debug_checks.py   : --debug sanity checks
#   - evaluator.py      : shared val-first / test-once driver
# ============================================================
