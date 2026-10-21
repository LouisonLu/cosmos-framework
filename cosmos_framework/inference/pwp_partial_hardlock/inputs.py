# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Media loading and batch construction for partial-hardlock no-video-path inference.

The batch follows the PwP training contract: three vision items
``[masked first RGB frame, full PCD video, generated RGB placeholder]``; all but
the last are clean conditions and the last is fully noised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.local_datasets.helper import ffmpeg_decode_video, get_video_metadata
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.pwp_partial_hardlock.config import PartialHardlockRequest
from cosmos_framework.inference.pwp_partial_hardlock.latent_hardlock import erode_keep_mask
from cosmos_framework.inference.structured_caption import caption_json_to_prompt


@dataclass(frozen=True)
class PreparedSample:
    batch: dict[str, Any]
    first_frame: np.ndarray  # [H,W,3] uint8, as given in image_context_path
    masked_first_frame: np.ndarray  # [H,W,3] uint8, the model's first vision item
    keep_mask: np.ndarray  # [H,W] bool, masks the model's first frame
    hardlock_mask: np.ndarray  # [H,W] bool, eroded keep mask used for the latent hardlock
    pcd_frames: np.ndarray  # [T,H,W,3] uint8
    fps: float


def load_prompt(path: str | Path) -> str:
    """Read a plain-text prompt, a ``{"caption": ...}`` JSON, or a structured caption JSON."""
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() != ".json":
        return text
    raw = json.loads(text)
    if isinstance(raw, dict):
        for key in ("caption", "text", "prompt"):
            if isinstance(raw.get(key), str) and raw[key].strip():
                return raw[key].strip()
        if "caption_json" in raw:
            raw = raw["caption_json"]
    return caption_json_to_prompt(raw) if isinstance(raw, dict) else str(raw).strip()


def load_image(path: Path, *, height: int, width: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def _first_video_frame(path: Path) -> np.ndarray:
    frames = ffmpeg_decode_video(str(path), scale_hw=None, num_threads=1)
    try:
        return next(frames)
    except StopIteration as error:
        raise ValueError(f"Video has no frames: {path}") from error
    finally:
        frames.close()


def load_keep_mask(path: Path | None, *, height: int, width: int) -> np.ndarray:
    """Load frame 0 of a keep mask as ``[H,W]`` bool; ``None`` keeps everything."""
    if path is None:
        return np.ones((height, width), dtype=bool)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            if "arr_0" not in data:
                raise ValueError(f"NPZ mask must contain 'arr_0' with shape [T,H,W]: {path}")
            binary = np.asarray(data["arr_0"][0]) > 0
    elif suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        binary = _first_video_frame(path).mean(axis=-1) >= 128
    else:
        with Image.open(path) as image:
            binary = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    if binary.shape != (height, width):
        resized = Image.fromarray(binary.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
        binary = np.asarray(resized, dtype=np.uint8) > 0
    if not binary.any():
        raise ValueError(f"Keep mask has no white pixels: {path}")
    return binary


def load_pcd(
    path: Path, *, height: int, width: int, num_frames: int, temporal_compression: int
) -> tuple[np.ndarray, float]:
    """Read up to ``num_frames`` PCD frames, trimmed to ``1 + k * temporal_compression``."""
    info = get_video_metadata(str(path))
    frames: list[np.ndarray] = []
    decoder = ffmpeg_decode_video(str(path), scale_hw=(height, width), num_threads=2)
    try:
        for frame in decoder:
            frames.append(frame)
            if len(frames) == num_frames:
                break
    finally:
        decoder.close()
    if not frames:
        raise ValueError(f"PCD video has no frames: {path}")
    usable = 1 + (len(frames) - 1) // temporal_compression * temporal_compression
    return np.stack(frames[:usable]), float(info["fps"])


def _normalize(frames: np.ndarray) -> torch.Tensor:
    """uint8 ``[T,H,W,3]`` -> float32 ``[3,T,H,W]`` in [-1,1]."""
    return torch.from_numpy(np.ascontiguousarray(frames)).float().permute(3, 0, 1, 2) / 127.5 - 1.0


def build_batch(
    *,
    masked_first_frame: np.ndarray,
    pcd_frames: np.ndarray,
    prompt: str,
    prompt_key: str,
    fps: float,
    device: str | torch.device,
) -> dict[str, Any]:
    height, width = masked_first_frame.shape[:2]
    if pcd_frames.shape[1:] != masked_first_frame.shape:
        raise ValueError(f"First frame/PCD shape mismatch: {masked_first_frame.shape} vs {pcd_frames.shape}")
    tensors = [
        _normalize(masked_first_frame[None]),
        _normalize(pcd_frames),
        _normalize(np.zeros_like(pcd_frames)),  # fully noised target; pixels only carry its shape
    ]
    size = torch.tensor([[height, width, height, width]], dtype=torch.float32, device=device)
    return {
        "dataset_name": "cosmos3_pwp_partial_hardlock_inference",
        "video": [tensor.unsqueeze(0).to(device=device) for tensor in tensors],
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
                vision_temporal_position_groups=[None, 0, 0],
            )
        ],
    }


def prepare_sample(
    request: PartialHardlockRequest,
    *,
    prompt_key: str,
    temporal_compression: int,
    device: str | torch.device,
) -> PreparedSample:
    first_frame = load_image(request.image_context_path, height=request.height, width=request.width)
    keep_mask = load_keep_mask(request.guided_generation_mask, height=request.height, width=request.width)
    hardlock_mask = erode_keep_mask(keep_mask, request.guided_generation_mask_erode_px)
    if not hardlock_mask.any():
        raise ValueError(
            f"Eroding the keep mask by {request.guided_generation_mask_erode_px}px leaves nothing to hardlock: "
            f"{request.guided_generation_mask}"
        )
    masked_first_frame = np.full_like(first_frame, request.mask_fill_value)
    masked_first_frame[keep_mask] = first_frame[keep_mask]
    pcd_frames, pcd_fps = load_pcd(
        request.pcd_path,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        temporal_compression=temporal_compression,
    )
    fps = pcd_fps if request.fps is None else request.fps
    batch = build_batch(
        masked_first_frame=masked_first_frame,
        pcd_frames=pcd_frames,
        prompt=request.prompt,
        prompt_key=prompt_key,
        fps=fps,
        device=device,
    )
    return PreparedSample(
        batch=batch,
        first_frame=first_frame,
        masked_first_frame=masked_first_frame,
        keep_mask=keep_mask,
        hardlock_mask=hardlock_mask,
        pcd_frames=pcd_frames,
        fps=fps,
    )
