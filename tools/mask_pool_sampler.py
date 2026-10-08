#!/usr/bin/env python3
"""Weighted DB235 mask-pool sampler for Cosmos3 PwP Stage1.

The sampler matches the Cosmos2.5 Stage1 mask-pool policy:

* AV2 masks: 60 percent
* native FOV, viewdrop, renderer confidence, and mixed: 10 percent each
* AV2 buckets: one third each for 10-15, 15-20, and 20-25 percent masks

One path is sampled for each training sample. The dataset applies that mask to
the first RGB frame and every PCD control frame; the RGB target is unchanged.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_FAMILY_WEIGHTS = {
    "av2": 60.0,
    "native_fov": 10.0,
    "viewdrop": 10.0,
    "renderer_confidence": 10.0,
    "mixed": 10.0,
}

DEFAULT_AV2_BUCKET_WEIGHTS = {
    "mask_10_15_percent": 20.0,
    "mask_15_20_percent": 20.0,
    "mask_20_25_percent": 20.0,
}


def _normalize_weights(weights: dict[str, float], available_keys: set[str]) -> dict[str, float]:
    filtered = {key: float(value) for key, value in weights.items() if key in available_keys and float(value) > 0}
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError(f"No positive weights remain. requested={weights}, available={sorted(available_keys)}")
    return {key: value / total for key, value in filtered.items()}


class MaskPoolSampler:
    """Sample masks from a DB235 train/val folder layout.

    Accepted roots are either ``root/{train,val}/{family}`` or directly
    ``root/{family}``. Non-AV2 files are uniform within the selected family;
    AV2 files are uniform within the selected bucket.
    """

    def __init__(
        self,
        pool_root: str | Path,
        seed: int = 0,
        split: str = "train",
        family_weights: dict[str, float] | None = None,
        av2_bucket_weights: dict[str, float] | None = None,
    ) -> None:
        self.root = Path(pool_root)
        self.split = split
        self.rng = random.Random(seed)

        self.split_root = self._resolve_split_root(self.root, split)
        self.family_dirs = {path.name: path for path in self.split_root.iterdir() if path.is_dir()}
        if not self.family_dirs:
            raise FileNotFoundError(f"No family directories found under {self.split_root}")

        requested_family_weights = dict(family_weights or DEFAULT_FAMILY_WEIGHTS)
        self.family_weights = _normalize_weights(requested_family_weights, set(self.family_dirs))

        self.family_pngs: dict[str, list[Path]] = {}
        for family, family_dir in self.family_dirs.items():
            if family == "av2":
                continue
            pngs = sorted(family_dir.rglob("*.png"))
            if pngs:
                self.family_pngs[family] = pngs

        self.av2_bucket_pngs: dict[str, list[Path]] = {}
        if "av2" in self.family_dirs:
            av2_dir = self.family_dirs["av2"]
            for bucket_dir in sorted(path for path in av2_dir.iterdir() if path.is_dir()):
                pngs = sorted(bucket_dir.rglob("*.png"))
                if pngs:
                    self.av2_bucket_pngs[bucket_dir.name] = pngs

        if "av2" in self.family_weights and not self.av2_bucket_pngs:
            raise ValueError(
                f"Family weight includes av2, but no AV2 bucket masks found under {self.family_dirs['av2']}"
            )

        requested_bucket_weights = dict(av2_bucket_weights or DEFAULT_AV2_BUCKET_WEIGHTS)
        self.av2_bucket_weights = (
            _normalize_weights(requested_bucket_weights, set(self.av2_bucket_pngs))
            if self.av2_bucket_pngs
            else {}
        )
        self._validate_non_av2_families()

    @staticmethod
    def _resolve_split_root(root: Path, split: str) -> Path:
        if (root / "av2").is_dir() or (root / "mixed").is_dir():
            return root
        candidate = root / split
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"Could not resolve split root from pool_root={root} and split={split}. "
            f"Expected family dirs directly under root or root/{split}."
        )

    def _validate_non_av2_families(self) -> None:
        missing = [
            family for family in self.family_weights if family != "av2" and not self.family_pngs.get(family)
        ]
        if missing:
            raise ValueError(
                f"Requested family weights include families with no PNG files: {missing}. "
                f"Available non-empty families: {sorted(key for key, value in self.family_pngs.items() if value)}"
            )

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        keys = list(weights)
        return self.rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]

    def sample_path(self) -> tuple[Path, dict[str, Any]]:
        family = self._weighted_choice(self.family_weights)
        if family == "av2":
            bucket = self._weighted_choice(self.av2_bucket_weights)
            path = self.rng.choice(self.av2_bucket_pngs[bucket])
            return path, {
                "family": family,
                "bucket": bucket,
                "path": str(path),
                "split": self.split_root.name,
                "sampling": "weighted_family_then_bucket_uniform_file",
            }

        path = self.rng.choice(self.family_pngs[family])
        return path, {
            "family": family,
            "bucket": None,
            "path": str(path),
            "split": self.split_root.name,
            "sampling": "weighted_family_uniform_file",
        }

    def sample(self) -> tuple[np.ndarray, dict[str, Any]]:
        path, metadata = self.sample_path()
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        values = set(np.unique(mask).tolist())
        if mask.shape != (512, 1024) or not values.issubset({0, 255}):
            raise ValueError(f"Invalid mask contract: {path}, shape={mask.shape}, values={sorted(values)}")
        return mask, metadata
