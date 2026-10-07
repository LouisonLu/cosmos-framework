# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Mask-pool helpers for the isolated PwP Cosmos3 Stage1 dataset.

The pool stores binary *keep* masks: white pixels are visible input and black
pixels are replaced by ``fill_value``.  One mask is sampled per sample and the
same resized mask is applied to the first RGB frame and every PCD control
frame.  The RGB target remains full-frame.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import get_worker_info


class MaskPool:
    """Load static or legacy-compatible dynamic masks from a pool."""

    def __init__(
        self,
        root: str | None = None,
        *,
        sampler_py: str | None = None,
        sampler_seed: int = 0,
        sampler_kwargs: str | dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).expanduser() if root else None
        self.sampler_py = str(Path(sampler_py).expanduser()) if sampler_py else None
        self.sampler_seed = int(sampler_seed)
        self.sampler_kwargs = self._parse_kwargs(sampler_kwargs)
        self._dynamic_sampler: Any | None = None
        self._dynamic_sampler_worker_id: int | None = None
        self._paths = sorted(self.root.rglob("*.png")) if self.root else []
        if self.root is not None and not self.root.is_dir():
            raise FileNotFoundError(f"Mask pool root not found: {self.root}")
        if self.sampler_py and not Path(self.sampler_py).is_file():
            raise FileNotFoundError(f"Mask sampler not found: {self.sampler_py}")
        if self.root is not None and not self.sampler_py and not self._paths:
            raise FileNotFoundError(f"No PNG masks found under {self.root}")

    @staticmethod
    def _parse_kwargs(raw: str | dict[str, Any] | None) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        text = raw.strip()
        if not text:
            return {}
        path = Path(text)
        value = json.loads(path.read_text() if path.is_file() else text)
        if not isinstance(value, dict):
            raise ValueError("mask_pool_sampler_kwargs must be a JSON object")
        return value

    def _load_dynamic_sampler(self) -> Any:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        if self._dynamic_sampler is not None and self._dynamic_sampler_worker_id == worker_id:
            return self._dynamic_sampler
        if self.sampler_py is None or self.root is None:
            raise ValueError("A dynamic mask sampler requires sampler_py and root")
        spec = importlib.util.spec_from_file_location("pwp_stage1_dynamic_mask_sampler", self.sampler_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import mask sampler: {self.sampler_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sampler_cls = getattr(module, "MaskPoolSampler", None) or getattr(module, "MaskPoolSamplerV11", None)
        if sampler_cls is None:
            raise AttributeError("Mask sampler must define MaskPoolSampler or MaskPoolSamplerV11")
        self._dynamic_sampler = sampler_cls(self.root, seed=self.sampler_seed + worker_id, **self.sampler_kwargs)
        self._dynamic_sampler_worker_id = worker_id
        return self._dynamic_sampler

    def sample(self) -> np.ndarray | None:
        """Return one uint8 binary keep mask, or ``None`` when no pool is configured."""
        if self.root is None:
            return None
        if self.sampler_py:
            result = self._load_dynamic_sampler().sample()
            mask = result[0] if isinstance(result, tuple) else result
        else:
            if not self._paths:
                raise ValueError("Mask pool is empty")
            # NumPy's process-local RNG follows the framework worker seed.
            with Image.open(self._paths[np.random.randint(len(self._paths))]) as image:
                mask = np.asarray(image.convert("L"))
        mask = np.asarray(mask, dtype=np.uint8)
        values = set(np.unique(mask).tolist())
        if mask.ndim != 2 or not values.issubset({0, 255}):
            raise ValueError(f"Mask must be binary uint8 [0,255], got shape={mask.shape}, values={sorted(values)}")
        return mask


def resize_keep_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a binary keep mask with nearest-neighbor semantics."""
    if mask.shape == (height, width):
        return mask.astype(bool, copy=False)
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def apply_keep_mask(frames: np.ndarray, keep_mask: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Fill masked pixels in ``[H,W,3]`` or ``[T,H,W,3]`` uint8 frames."""
    if frames.ndim == 3:
        output = np.full_like(frames, np.clip(fill_value, 0, 255))
        output[keep_mask] = frames[keep_mask]
    elif frames.ndim == 4:
        output = np.full_like(frames, np.clip(fill_value, 0, 255))
        output[:, keep_mask] = frames[:, keep_mask]
    else:
        raise ValueError(f"Expected [H,W,3] or [T,H,W,3], got {frames.shape}")
    return output
