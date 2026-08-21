from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def next_experiment_dir(root: str | Path, prefix: str, name: str) -> Path:
    root_path = ensure_dir(root)
    existing = sorted(root_path.glob(f"{prefix}[0-9][0-9][0-9]_{name}"))
    used = []
    for path in existing:
        stem = path.name.split("_", 1)[0]
        number = stem.replace(prefix, "")
        if number.isdigit():
            used.append(int(number))
    next_id = (max(used) + 1) if used else 1
    exp_dir = root_path / f"{prefix}{next_id:03d}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir

