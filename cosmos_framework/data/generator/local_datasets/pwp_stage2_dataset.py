# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""PwP Stage 2 local dataset for Cosmos3.

Each record provides a paired per-video validity mask.  The sample emitted to
Cosmos3 is:

    [masked RGB first frame, full PCD control video, masked RGB target video]

The mask is also emitted as ``target_loss_mask``.  The model consumes that
tensor at the latent resolution and excludes black pixels from the target
flow-matching loss.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from cosmos_framework.data.generator.local_datasets.helper import (
    ffmpeg_decode_video,
    get_video_metadata,
)
from cosmos_framework.data.generator.local_datasets.pwp_stage1_dataset import (
    _load_jsonl,
    _normalize,
    _prepare_metadata,
    _resolve_path,
)
from cosmos_framework.data.generator.local_datasets.sft_dataset import SFTDataset
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.utils import log


def _prepare_stage2_metadata(records: list[dict[str, Any]], manifest_path: str) -> list[dict[str, Any]]:
    """Resolve the Stage 2 mask path while preserving the Stage 1 manifest schema."""
    prepared = _prepare_metadata(records, manifest_path)
    for raw, item in zip(records, prepared, strict=True):
        if item.get("mask_path") is None:
            for alias in ("target_loss_mask_path", "mask_video_path"):
                if raw.get(alias):
                    item["mask_path"] = _resolve_path(str(raw[alias]), manifest_path)
                    break
        if not item.get("mask_path"):
            raise ValueError(
                f"Record {item['uuid']} needs mask_path or target_loss_mask_path for PwP Stage 2"
            )
        if not Path(item["mask_path"]).is_file():
            raise FileNotFoundError(f"Stage 2 mask not found for {item['uuid']}: {item['mask_path']}")
    return prepared


class PwPStage2Dataset(SFTDataset):
    """Load masked RGB targets, full PCD conditions, prompts, and loss masks."""

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
        mask_threshold: int = 128,
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
        if not 0 <= mask_threshold <= 255:
            raise ValueError("mask_threshold must be in [0, 255]")

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
        self.mask_threshold = int(mask_threshold)
        self.mask_fill_value = int(mask_fill_value)

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
            raise ValueError("start_frame is beyond the input videos")
        interval = max(1, int(fps / 30.0)) if self.temporal_interval_mode == "max_30fps" else 1
        count = available if requested == -1 else min(requested, available // interval + (1 if available else 0))
        if count < 1:
            raise ValueError("No frames available after temporal sampling")
        span = (count - 1) * interval + 1
        if self.frame_selection_mode == "center":
            start += max(0, (available - span) // 2)
        elif self.frame_selection_mode == "random":
            start += np.random.randint(max(1, available - span + 1))
        return start, start + span - 1, interval

    def process_one_sample(self, metadata: dict[str, Any]) -> dict | None:
        try:
            rgb_frames, rgb_info = self._decode(metadata["rgb_path"], self.target_height, self.target_width)
            pcd_frames, _ = self._decode(metadata["pcd_path"], self.target_height, self.target_width)
            mask_frames, _ = self._decode(metadata["mask_path"], self.target_height, self.target_width)
            available_frames = min(len(rgb_frames), len(pcd_frames), len(mask_frames))
            start, end, interval = self._select_frames(
                available_frames,
                rgb_info["fps"],
                self.num_video_frames,
                metadata,
            )
            frame_ids = list(range(start, end + 1, interval))
            rgb = rgb_frames[frame_ids]
            pcd = pcd_frames[frame_ids]
            mask = mask_frames[frame_ids, ..., 0] >= self.mask_threshold
            if len(rgb) != len(pcd) or len(rgb) != len(mask):
                raise ValueError(
                    f"RGB/PCD/mask frame mismatch: {len(rgb)} vs {len(pcd)} vs {len(mask)}"
                )

            usable_t = 1 + (len(rgb) - 1) // self.temporal_compression_factor * self.temporal_compression_factor
            rgb = rgb[:usable_t]
            pcd = pcd[:usable_t]
            mask = mask[:usable_t]

            masked_rgb = np.full_like(rgb, np.clip(self.mask_fill_value, 0, 255))
            masked_rgb[mask] = rgb[mask]
            target_loss_mask = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0)

            caption = metadata["prompt"]
            if self.caption_suffix:
                caption = f"{caption} {self.caption_suffix}".strip()
            if self.cfg_dropout_rate > 0 and np.random.random() < self.cfg_dropout_rate:
                caption = ""
            text_ids, caption = self._tokenize_caption(caption)
            image_size = torch.tensor(
                [self.target_height, self.target_width, self.target_height, self.target_width], dtype=torch.float32
            )
            return {
                "__key__": f"{metadata['uuid']}_stage2",
                "__url__": metadata["rgb_path"],
                "fps": float(rgb_info["fps"]),
                "n_orig_video_frames": int(rgb_info["total_frames"]),
                "chunk_index": 0,
                "frame_start": start,
                "frame_end": end,
                "num_frames": len(masked_rgb),
                "video": [_normalize(masked_rgb[:1]), _normalize(pcd), _normalize(masked_rgb)],
                "target_loss_mask": target_loss_mask,
                "num_multiplier": interval,
                "conditioning_fps": float(rgb_info["fps"] if self.conditioning_fps < 0 else self.conditioning_fps),
                "image_size": [image_size, image_size.clone(), image_size.clone()],
                "ai_caption": caption,
                "selected_caption_type": "pwp_stage2_prompt",
                "text_token_ids": torch.tensor(text_ids),
                "dataset_name": "pwp_stage2_target_loss_mask",
                "sequence_plan": SequencePlan(
                    has_text=True,
                    has_vision=True,
                    condition_frame_indexes_vision=[],
                    share_vision_temporal_positions=False,
                    vision_temporal_position_groups=[None, 0, 0],
                ),
            }
        except Exception as exc:
            log.warning(f"PwP Stage 2 sample failed for {metadata.get('uuid')}: {exc}", rank0_only=False)
            return None


def get_pwp_stage2_dataset(
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
    mask_threshold: int = 128,
    mask_fill_value: int = 0,
    **kwargs: Any,
) -> PwPStage2Dataset:
    """Build the independent Cosmos3 Stage 2 dataset from JSONL manifests."""
    del kwargs
    paths = [jsonl_paths] if isinstance(jsonl_paths, str) else list(jsonl_paths)
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_prepare_stage2_metadata(_load_jsonl(path), path))
    return PwPStage2Dataset(
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
        mask_threshold=mask_threshold,
        mask_fill_value=mask_fill_value,
    )


__all__ = ["PwPStage2Dataset", "get_pwp_stage2_dataset"]
