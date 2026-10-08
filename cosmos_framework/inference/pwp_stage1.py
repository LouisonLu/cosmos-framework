# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Independent inference input builder for the Cosmos3 PwP Stage1 recipe.

Stage1 is a three-item vision sample:

    [masked first-frame RGB, masked PCD control video, generated RGB video]

The first two items are clean controls and the last item is the only generated
item.  This module deliberately mirrors ``pwp_stage1_dataset.py`` instead of
using the stock one-video ``get_sample_data`` helper.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.local_datasets.helper import (
    ffmpeg_decode_video,
    get_video_metadata,
)
from cosmos_framework.data.generator.local_datasets.pwp_stage1_mask_pool import (
    apply_keep_mask,
    resize_keep_mask,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.structured_caption import caption_json_to_prompt


@dataclass(frozen=True)
class PwPStage1PreparedInputs:
    """Decoded media kept alongside the inference batch for optional hardlock."""

    batch: dict[str, Any]
    first_frame: np.ndarray
    pcd_frames: np.ndarray
    keep_mask: np.ndarray
    fps: float
    height: int
    width: int


def normalize_frames(frames: np.ndarray) -> torch.Tensor:
    """Convert uint8 THWC pixels to tokenizer-native float32 CTHW in [-1, 1]."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected uint8 [T,H,W,3] frames, got {frames.dtype} {frames.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float()
    return tensor.permute(3, 0, 1, 2) / 127.5 - 1.0


def decode_video(path: str | Path, *, height: int, width: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode a local video using the same ffmpeg path as the training dataset."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with tempfile.NamedTemporaryFile(suffix=path.suffix or ".mp4") as tmp:
        tmp.write(path.read_bytes())
        tmp.flush()
        info = get_video_metadata(tmp.name)
        frames = np.stack(list(ffmpeg_decode_video(tmp.name, scale_hw=(height, width), num_threads=2)))
    return frames, info


def decode_image(path: str | Path, *, height: int, width: int) -> np.ndarray:
    """Decode one RGB image and resize it to the training resolution."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def build_pwp_stage1_batch(
    *,
    first_frame: np.ndarray | None = None,
    pcd_frames: np.ndarray,
    keep_mask: np.ndarray,
    prompt: str,
    prompt_key: str,
    fps: float,
    device: str | torch.device = "cuda",
    mask_fill_value: int = 0,
    temporal_position_groups: list[int | None] | None = None,
) -> dict[str, Any]:
    """Build the flattened multi-vision batch expected by inference.

    ``first_frame`` and ``pcd_frames`` are already aligned and resized.  The
    generated third item is a zero placeholder: with an empty vision
    condition-frame list it is fully noisy, so its pixels are used only to
    carry the target shape through the VAE interface.

    """
    if first_frame is None:
        raise ValueError("first_frame is required")
    if first_frame.ndim == 3:
        first_frame = first_frame[None]
    if first_frame.ndim != 4 or first_frame.shape[0] != 1 or pcd_frames.ndim != 4:
        raise ValueError("first_frame must be [1,H,W,3] and PCD must be [T,H,W,3]")
    if first_frame.shape[1:] != pcd_frames.shape[1:]:
        raise ValueError(f"First frame/PCD shape mismatch: {first_frame.shape} vs {pcd_frames.shape}")
    if keep_mask.shape != first_frame.shape[1:3]:
        raise ValueError(
            f"Mask shape {keep_mask.shape} does not match video {(first_frame.shape[1], first_frame.shape[2])}"
        )
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    masked_first = apply_keep_mask(first_frame, keep_mask, mask_fill_value)
    masked_pcd = apply_keep_mask(pcd_frames, keep_mask, mask_fill_value)
    generated_placeholder = np.zeros_like(pcd_frames)

    tensors = [
        normalize_frames(masked_first),
        normalize_frames(masked_pcd),
        normalize_frames(generated_placeholder),
    ]
    height, width = first_frame.shape[1:3]
    size = torch.tensor([[height, width, height, width]], dtype=torch.float32, device=device)
    groups = temporal_position_groups or [None, 0, 0]
    if len(groups) != 3:
        raise ValueError(f"temporal_position_groups must have 3 entries, got {groups}")

    # In direct inference the multi-item list is already flattened, as in the
    # official transfer builder. Training starts nested and the dataloader
    # flattens it before reaching the model.
    video = [tensor.unsqueeze(0).to(device=device) for tensor in tensors]
    return {
        "dataset_name": "pwp_stage1_inference",
        "video": video,
        "image_size": [size, size.clone(), size.clone()],
        "num_vision_items_per_sample": [3],
        "num_frames": torch.tensor([pcd_frames.shape[0]], dtype=torch.int64, device=device),
        "fps": torch.tensor([fps], dtype=torch.float32, device=device),
        "conditioning_fps": torch.tensor([fps], dtype=torch.float32, device=device),
        "is_preprocessed": True,
        "system_prompt": None,
        prompt_key: [prompt],
        "sequence_plan": [
            SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[],
                share_vision_temporal_positions=False,
                vision_temporal_position_groups=groups,
            )
        ],
    }


def load_prompt(path: str | Path) -> str:
    """Load a plain prompt or structured caption with training-identical JSON serialization."""
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() != ".json":
        return text
    raw = json.loads(text)
    if isinstance(raw, dict) and isinstance(raw.get("caption"), str):
        return raw["caption"].strip()
    if isinstance(raw, dict) and "caption_json" in raw:
        raw = raw["caption_json"]
    return caption_json_to_prompt(raw) if isinstance(raw, dict) else str(raw).strip()


def load_pwp_stage1_inputs(
    *,
    first_frame_path: str | Path,
    pcd_path: str | Path,
    mask_path: str | Path | None,
    prompt: str,
    prompt_key: str,
    height: int = 512,
    width: int = 1024,
    num_frames: int = 93,
    start_frame: int = 0,
    temporal_interval_mode: str = "force_one",
    temporal_compression_factor: int = 4,
    fps: float | None = None,
    device: str | torch.device = "cuda",
    mask_fill_value: int = 0,
) -> PwPStage1PreparedInputs:
    """Decode, align, mask, and package one Stage1 inference sample.

    Unlike the generic video-to-video route, this function needs no RGB video:
    the PCD video supplies the generated clip length and the RGB input is one
    first-frame image.
    """
    first_frame = decode_image(first_frame_path, height=height, width=width)
    pcd_all, pcd_info = decode_video(pcd_path, height=height, width=width)
    if temporal_interval_mode not in {"force_one", "max_30fps"}:
        raise ValueError(f"Unsupported temporal_interval_mode={temporal_interval_mode!r}")
    if start_frame < 0 or start_frame >= len(pcd_all):
        raise ValueError(f"start_frame={start_frame} is outside the PCD video with {len(pcd_all)} frames")
    interval = max(1, int(float(pcd_info["fps"]) / 30.0)) if temporal_interval_mode == "max_30fps" else 1
    available = len(pcd_all) - start_frame
    count = available if num_frames == -1 else min(num_frames, available // interval + 1)
    if count < 1:
        raise ValueError("No PCD frames available after temporal sampling")
    frame_ids = list(range(start_frame, start_frame + (count - 1) * interval + 1, interval))
    usable_t = 1 + (len(frame_ids) - 1) // temporal_compression_factor * temporal_compression_factor
    pcd = pcd_all[np.asarray(frame_ids[:usable_t], dtype=np.int64)]

    if mask_path is None:
        keep_mask = np.ones((height, width), dtype=bool)
    else:
        with Image.open(Path(mask_path).expanduser()) as mask_image:
            raw_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
        keep_mask = resize_keep_mask(raw_mask, height, width)

    effective_fps = float(pcd_info["fps"] if fps is None else fps)
    batch = build_pwp_stage1_batch(
        first_frame=first_frame,
        pcd_frames=pcd,
        keep_mask=keep_mask,
        prompt=prompt,
        prompt_key=prompt_key,
        fps=effective_fps,
        device=device,
        mask_fill_value=mask_fill_value,
    )
    return PwPStage1PreparedInputs(
        batch=batch,
        first_frame=first_frame,
        pcd_frames=pcd,
        keep_mask=keep_mask,
        fps=effective_fps,
        height=height,
        width=width,
    )


def hardlock_first_frame(
    decoded_video: torch.Tensor,
    *,
    rgb_first_frame: np.ndarray,
    keep_mask: np.ndarray,
) -> torch.Tensor:
    """Blend the original RGB into the white keep region of decoded frame zero.

    This is the decode-space counterpart of Cosmos2.5's first-frame hardlock;
    it is intentionally optional because it is not a Cosmos3 latent-space
    condition mask.
    """
    if decoded_video.ndim != 5 or decoded_video.shape[0] != 1:
        raise ValueError(f"Expected decoded video [1,3,T,H,W], got {tuple(decoded_video.shape)}")
    if rgb_first_frame.shape != (*keep_mask.shape, 3):
        raise ValueError("rgb_first_frame and keep_mask shapes do not match")
    reference = torch.from_numpy(np.ascontiguousarray(rgb_first_frame)).float().permute(2, 0, 1) / 255.0
    reference = reference.to(device=decoded_video.device, dtype=decoded_video.dtype).unsqueeze(0)
    mask = torch.from_numpy(np.ascontiguousarray(keep_mask)).to(device=decoded_video.device)
    mask = mask.to(dtype=decoded_video.dtype).view(1, 1, *keep_mask.shape)
    output = decoded_video.clone()
    output[:, :, 0] = mask * reference + (1.0 - mask) * output[:, :, 0]
    return output
