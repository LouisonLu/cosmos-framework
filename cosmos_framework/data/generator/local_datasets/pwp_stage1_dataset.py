# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""PwP Stage1 local dataset for Cosmos3.

Each sample is packed as three vision items:

    [masked RGB first-frame image, full PCD control video, full RGB target video]

The first two items are automatically treated as fully-clean controls by the
Cosmos3 sequence packer.  The last item has an empty conditioning-frame list,
so its complete RGB target latent is used by the standard flow-matching loss.
No target-validity mask is emitted; that belongs to the future Stage2 pipeline.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.local_datasets.helper import (
    ffmpeg_decode_video,
    get_video_metadata,
)
from cosmos_framework.data.generator.local_datasets.pwp_stage1_mask_pool import (
    MaskPool,
    apply_keep_mask,
    resize_keep_mask,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import SFTDataset
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.structured_caption import caption_json_to_prompt
from cosmos_framework.utils import log


def _normalize(frames: np.ndarray) -> torch.Tensor:
    """Convert uint8 THWC frames to tokenizer-native float32 CTHW in [-1,1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float()
    return tensor.permute(3, 0, 1, 2) / 127.5 - 1.0


def _resolve_path(value: str, manifest_path: str) -> str:
    if "://" in value or os.path.isabs(value):
        return value
    return str((Path(manifest_path).parent / value).resolve())


def _load_prompt(record: dict[str, Any], manifest_path: str) -> str:
    raw: Any = record.get("prompt", record.get("caption"))
    prompt_path = record.get("prompt_path", record.get("caption_path"))
    if raw is None and prompt_path is None:
        raise ValueError("Manifest record needs prompt, caption, prompt_path, or caption_path")
    if prompt_path is not None:
        path = _resolve_path(str(prompt_path), manifest_path)
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "caption" in raw and isinstance(raw["caption"], str):
            return raw["caption"].strip()
        if "caption_json" in raw:
            raw = raw["caption_json"]
        return caption_json_to_prompt(raw) if isinstance(raw, dict) else str(raw).strip()
    return str(raw).strip()


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    data = Path(path).read_bytes() if not path.startswith("s3://") else None
    if data is None:
        raise ValueError("PwP Stage1 currently expects a local JSONL manifest")
    stream = gzip.GzipFile(fileobj=io.BytesIO(data)) if path.endswith(".gz") else io.BytesIO(data)
    records = [json.loads(line.decode("utf-8")) for line in stream if line.strip()]
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _prepare_metadata(records: list[dict[str, Any]], manifest_path: str) -> list[dict[str, Any]]:
    prepared = []
    for index, record in enumerate(records):
        rgb = record.get("rgb_path", record.get("vision_path"))
        pcd = record.get("pcd_path", record.get("control_path"))
        if not rgb or not pcd:
            raise ValueError(f"Record {index} must contain rgb_path/vision_path and pcd_path/control_path")
        rgb = _resolve_path(str(rgb), manifest_path)
        pcd = _resolve_path(str(pcd), manifest_path)
        uuid = str(record.get("uuid", Path(rgb).stem))
        prompt = _load_prompt(record, manifest_path)
        prepared.append(
            {
                **record,
                "uuid": uuid,
                "vision_path": rgb,
                "rgb_path": rgb,
                "pcd_path": pcd,
                "prompt": prompt,
                "mask_path": (
                    _resolve_path(str(record["mask_path"]), manifest_path) if record.get("mask_path") else None
                ),
            }
        )
    return prepared


class PwPStage1Dataset(SFTDataset):
    """Iterable local dataset implementing the PwP Stage1 contract."""

    def __init__(
        self,
        *,
        metadata: list[dict[str, Any]],
        manifest_path: str,
        num_video_frames: int = 93,
        target_width: int = 1024,
        target_height: int = 512,
        frame_selection_mode: str = "first",
        temporal_interval_mode: str = "force_one",
        tokenizer_config: Optional[Any] = None,
        cfg_dropout_rate: float = 0.1,
        use_system_prompt: bool = False,
        max_caption_tokens: int = 2048,
        caption_suffix: str = "",
        conditioning_fps: float = -1,
        temporal_compression_factor: int = 4,
        mask_pool_root: str | None = None,
        mask_pool_sampler_py: str | None = None,
        mask_pool_sampler_seed: int = 0,
        mask_pool_sampler_kwargs_json: str | dict[str, Any] | None = None,
        mask_fill_value: int = 0,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if frame_selection_mode not in {"first", "center", "random"}:
            raise ValueError(f"Unsupported frame_selection_mode={frame_selection_mode!r}")
        if temporal_interval_mode not in {"force_one", "max_30fps"}:
            raise ValueError(f"Unsupported temporal_interval_mode={temporal_interval_mode!r}")
        if target_width < 32 or target_height < 32:
            raise ValueError("target_width and target_height must be at least 32")
        # The parent supplies the Cosmos tokenizer setup and rank-partitioned iterator.
        super().__init__(
            metadata=metadata,
            num_video_frames=num_video_frames,
            resolution="480",
            s3_credentials={},
            temporal_interval_mode="force_one",
            frame_selection_mode="first",
            tokenizer_config=tokenizer_config,
            cfg_dropout_rate=cfg_dropout_rate,
            use_system_prompt=use_system_prompt,
            max_caption_tokens=max_caption_tokens,
            append_duration_fps_timestamps=False,
            append_resolution_info=False,
            caption_suffix=caption_suffix,
            conditioning_fps=conditioning_fps,
            conditioning_config=None,
            temporal_compression_factor=temporal_compression_factor,
        )
        self.manifest_path = manifest_path
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.frame_selection_mode = frame_selection_mode
        self.temporal_interval_mode = temporal_interval_mode
        self.mask_fill_value = int(mask_fill_value)
        self.mask_pool = MaskPool(
            mask_pool_root,
            sampler_py=mask_pool_sampler_py,
            sampler_seed=mask_pool_sampler_seed,
            sampler_kwargs=mask_pool_sampler_kwargs_json,
        )

    @staticmethod
    def _decode(path: str, height: int, width: int) -> tuple[np.ndarray, dict[str, Any]]:
        with tempfile.NamedTemporaryFile(suffix=Path(path).suffix or ".mp4") as tmp:
            tmp.write(Path(path).read_bytes())
            tmp.flush()
            info = get_video_metadata(tmp.name)
            frames = np.stack(list(ffmpeg_decode_video(tmp.name, scale_hw=(height, width), num_threads=2)))
        return frames, info

    def _select_frames(
        self, total_frames: int, fps: float, requested: int, record: dict[str, Any]
    ) -> tuple[int, int, int]:
        start = int(record.get("start_frame", 0))
        available = total_frames - start
        if available < 1:
            raise ValueError("start_frame is beyond the RGB video")
        interval = max(1, int(fps / 30.0)) if self.temporal_interval_mode == "max_30fps" else 1
        count = available if requested == -1 else min(requested, available // interval + (1 if available else 0))
        if count < 1:
            raise ValueError("No frames available after temporal sampling")
        span = (count - 1) * interval + 1
        if self.frame_selection_mode == "center":
            start += max(0, (available - span) // 2)
        elif self.frame_selection_mode == "random":
            start += random.randint(0, max(0, available - span))
        return start, start + span - 1, interval

    def process_one_sample(self, metadata: dict[str, Any]) -> dict | None:
        try:
            rgb_frames, rgb_info = self._decode(metadata["rgb_path"], self.target_height, self.target_width)
            pcd_frames, pcd_info = self._decode(metadata["pcd_path"], self.target_height, self.target_width)
            start, end, interval = self._select_frames(
                min(len(rgb_frames), len(pcd_frames)),
                rgb_info["fps"],
                self.num_video_frames,
                metadata,
            )
            frame_ids = list(range(start, end + 1, interval))
            rgb = rgb_frames[frame_ids]
            pcd = pcd_frames[frame_ids]
            if len(rgb) != len(pcd):
                raise ValueError(f"RGB/PCD frame mismatch: {len(rgb)} vs {len(pcd)}")
            usable_t = 1 + (len(rgb) - 1) // self.temporal_compression_factor * self.temporal_compression_factor
            rgb = rgb[:usable_t]
            pcd = pcd[:usable_t]

            keep_mask = None
            if metadata.get("mask_path"):
                with Image.open(metadata["mask_path"]) as mask_image:
                    keep_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
            elif self.mask_pool.root is not None:
                keep_mask = self.mask_pool.sample()
            if keep_mask is None:
                keep_mask = np.full((self.target_height, self.target_width), 255, dtype=np.uint8)
            keep_mask = resize_keep_mask(keep_mask, self.target_height, self.target_width)

            masked_first = apply_keep_mask(rgb[:1], keep_mask, self.mask_fill_value)
            target = rgb

            caption = metadata["prompt"]
            if self.caption_suffix:
                caption = f"{caption} {self.caption_suffix}".strip()
            if self.cfg_dropout_rate > 0 and random.random() < self.cfg_dropout_rate:
                caption = ""
            text_ids, caption = self._tokenize_caption(caption)
            image_size = torch.tensor(
                [self.target_height, self.target_width, self.target_height, self.target_width], dtype=torch.float32
            )
            return {
                "__key__": f"{metadata['uuid']}_stage1",
                "__url__": metadata["rgb_path"],
                "fps": float(rgb_info["fps"]),
                "n_orig_video_frames": int(rgb_info["total_frames"]),
                "chunk_index": 0,
                "frame_start": start,
                "frame_end": end,
                "num_frames": len(target),
                "video": [_normalize(masked_first), _normalize(pcd), _normalize(target)],
                "num_multiplier": interval,
                "conditioning_fps": float(rgb_info["fps"] if self.conditioning_fps < 0 else self.conditioning_fps),
                "image_size": [image_size, image_size.clone(), image_size.clone()],
                "ai_caption": caption,
                "selected_caption_type": "pwp_stage1_prompt",
                "text_token_ids": torch.tensor(text_ids),
                "dataset_name": "pwp_stage1_transfer",
                "sequence_plan": SequencePlan(
                    has_text=True,
                    has_vision=True,
                    # Empty means every latent of the final RGB item is noisy and supervised.
                    condition_frame_indexes_vision=[],
                    share_vision_temporal_positions=False,
                    # The first RGB image is an independent reference. The PCD and RGB target
                    # are frame-aligned controls, so they share one temporal-position group.
                    vision_temporal_position_groups=[None, 0, 0],
                ),
            }
        except Exception as exc:
            log.warning(f"PwP Stage1 sample failed for {metadata.get('uuid')}: {exc}", rank0_only=False)
            return None


def get_pwp_stage1_dataset(
    jsonl_paths: str | list[str],
    tokenizer_config: Optional[Any] = None,
    num_video_frames: int = 93,
    target_width: int = 1024,
    target_height: int = 512,
    frame_selection_mode: str = "first",
    temporal_interval_mode: str = "force_one",
    cfg_dropout_rate: float = 0.1,
    use_system_prompt: bool = False,
    max_caption_tokens: int = 2048,
    caption_suffix: str = "",
    conditioning_fps: float = -1,
    temporal_compression_factor: int = 4,
    mask_pool_root: str | None = None,
    mask_pool_sampler_py: str | None = None,
    mask_pool_sampler_seed: int = 0,
    mask_pool_sampler_kwargs_json: str | dict[str, Any] | None = None,
    mask_fill_value: int = 0,
    **kwargs: Any,
) -> PwPStage1Dataset:
    """Build the local PwP Stage1 dataset from one or more JSONL manifests."""
    del kwargs
    paths = [jsonl_paths] if isinstance(jsonl_paths, str) else list(jsonl_paths)
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_prepare_metadata(_load_jsonl(path), path))
    return PwPStage1Dataset(
        metadata=records,
        manifest_path=paths[0],
        num_video_frames=num_video_frames,
        target_width=target_width,
        target_height=target_height,
        frame_selection_mode=frame_selection_mode,
        temporal_interval_mode=temporal_interval_mode,
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        use_system_prompt=use_system_prompt,
        max_caption_tokens=max_caption_tokens,
        caption_suffix=caption_suffix,
        conditioning_fps=conditioning_fps,
        temporal_compression_factor=temporal_compression_factor,
        mask_pool_root=mask_pool_root,
        mask_pool_sampler_py=mask_pool_sampler_py,
        mask_pool_sampler_seed=mask_pool_sampler_seed,
        mask_pool_sampler_kwargs_json=mask_pool_sampler_kwargs_json,
        mask_fill_value=mask_fill_value,
    )
