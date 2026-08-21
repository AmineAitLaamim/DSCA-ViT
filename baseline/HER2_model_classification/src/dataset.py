from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image_path: str
    label: str
    split: str
    patient_id: str = ""
    wsi_id: str = ""


def _iter_images(root: Path, extensions: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def scan_directory_dataset(data_dir: str | Path, config: dict[str, Any]) -> list[Sample]:
    root = Path(data_dir)
    dataset_config = config["dataset"]
    class_to_idx = dataset_config.get("class_mapping") or config["classes"]
    classes = list(class_to_idx.keys())
    extensions = set(config["dataset"]["allowed_extensions"])
    samples: list[Sample] = []
    split_dirs = {
        "train": dataset_config.get("train_dir", "train"),
        "val": dataset_config.get("val_dir", "val"),
        "test": dataset_config.get("test_dir", "test"),
    }
    required_splits = ("train", "test")
    for split in required_splits:
        split_dir = root / split_dirs[split]
        if not split_dir.exists():
            raise FileNotFoundError(f"Required {split} folder not found: {split_dir}")
        for label in classes:
            class_dir = split_dir / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Required class folder not found: {class_dir}")
            for image_path in _iter_images(class_dir, extensions):
                samples.append(Sample(str(image_path), label, split))
    val_dir_name = split_dirs.get("val")
    if val_dir_name:
        val_dir = root / val_dir_name
        if val_dir.exists():
            for label in classes:
                class_dir = val_dir / label
                if not class_dir.exists():
                    raise FileNotFoundError(f"Required validation class folder not found: {class_dir}")
                for image_path in _iter_images(class_dir, extensions):
                    samples.append(Sample(str(image_path), label, "val"))
    if not samples:
        raise FileNotFoundError(f"No dataset samples found under {root}")
    return samples


def read_manifest(path: str | Path) -> list[Sample]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "label", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
        return [
            Sample(
                image_path=row["image_path"],
                label=row["label"],
                split=row["split"],
                patient_id=row.get("patient_id", "") or "",
                wsi_id=row.get("wsi_id", "") or "",
            )
            for row in reader
        ]


def write_manifest(samples: list[Sample], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "split", "patient_id", "wsi_id"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def split_train_val(samples: list[Sample], val_fraction: float, seed: int) -> list[Sample]:
    if any(s.split == "val" for s in samples):
        return samples
    train = [s for s in samples if s.split == "train"]
    other = [s for s in samples if s.split != "train"]
    if not train:
        return samples
    labels = [s.label for s in train]
    train_part, val_part = train_test_split(
        train,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    return [Sample(s.image_path, s.label, "train", s.patient_id, s.wsi_id) for s in train_part] + [
        Sample(s.image_path, s.label, "val", s.patient_id, s.wsi_id) for s in val_part
    ] + other


def validate_samples(samples: list[Sample], config: dict[str, Any]) -> dict[str, Any]:
    class_to_idx = config["dataset"].get("class_mapping") or config["classes"]
    allowed = set(class_to_idx.keys())
    invalid = sorted({s.label for s in samples if s.label not in allowed})
    if invalid:
        raise ValueError(f"Invalid labels found: {invalid}")

    by_path: dict[str, set[str]] = {}
    for sample in samples:
        resolved = str(Path(sample.image_path).resolve())
        by_path.setdefault(resolved, set()).add(sample.split)
    leaked_paths = {p: sorted(splits) for p, splits in by_path.items() if len(splits) > 1}
    if leaked_paths:
        raise ValueError(f"Image path leakage across splits detected: {list(leaked_paths.items())[:5]}")

    for id_name in ("patient_id", "wsi_id"):
        ids_by_split: dict[str, set[str]] = {}
        for sample in samples:
            value = getattr(sample, id_name)
            if value:
                ids_by_split.setdefault(value, set()).add(sample.split)
        leaked_ids = {k: sorted(v) for k, v in ids_by_split.items() if len(v) > 1}
        if leaked_ids:
            raise ValueError(f"{id_name} leakage across splits detected: {list(leaked_ids.items())[:5]}")

    if config["dataset"].get("verify_images", False):
        for sample in samples:
            with Image.open(sample.image_path) as img:
                img.verify()

    if config["dataset"].get("verify_duplicate_content", False):
        digest_to_splits: dict[str, set[str]] = {}
        digest_to_path: dict[str, str] = {}
        for sample in samples:
            digest = file_digest(sample.image_path)
            digest_to_splits.setdefault(digest, set()).add(sample.split)
            digest_to_path.setdefault(digest, sample.image_path)
        leaked_content = {
            digest_to_path[digest]: sorted(splits)
            for digest, splits in digest_to_splits.items()
            if len(splits) > 1
        }
        if leaked_content:
            raise ValueError(f"Duplicate image content leakage across splits detected: {list(leaked_content.items())[:5]}")

    counts: dict[str, dict[str, int]] = {}
    for sample in samples:
        counts.setdefault(sample.split, {}).setdefault(sample.label, 0)
        counts[sample.split][sample.label] += 1
    effective_counts = {
        split: {label: counts.get(split, {}).get(label, 0) for label in class_to_idx}
        for split in ("train", "val", "test")
        if split in counts
    }
    return {"num_samples": len(samples), "class_counts": effective_counts}


def verify_official_counts(samples: list[Sample], config: dict[str, Any]) -> dict[str, Any]:
    expected = config["dataset"].get("expected_counts") or {}
    if not expected:
        return {"verified": False, "reason": "No expected_counts configured."}

    class_to_idx = config["dataset"].get("class_mapping") or config["classes"]
    actual: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        actual[split] = {label: 0 for label in class_to_idx}
    for sample in samples:
        if sample.split in actual:
            actual[sample.split][sample.label] += 1

    mismatches = []
    for split, expected_by_class in expected.items():
        for label, expected_count in expected_by_class.items():
            actual_count = actual.get(split, {}).get(label, 0)
            if actual_count != int(expected_count):
                mismatches.append(
                    f"{split}/{label}: expected {expected_count}, found {actual_count}"
                )
    if mismatches:
        raise ValueError("Official HER2-IHC-40x count verification failed: " + "; ".join(mismatches))
    return {"verified": True, "expected_counts": expected, "actual_counts": actual}


def file_digest(path: str | Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


class HER2PatchDataset(Dataset):
    def __init__(self, samples: list[Sample], class_to_idx: dict[str, int], transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[sample.label]
        return {
            "image": image,
            "label": label,
            "image_path": sample.image_path,
            "patient_id": sample.patient_id,
            "wsi_id": sample.wsi_id,
        }


def samples_for_split(samples: list[Sample], split: str) -> list[Sample]:
    return [s for s in samples if s.split == split]
