# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Input pairing and Cosmos3 PwP batch construction for mask-video guidance."""

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
from cosmos_framework.inference.structured_caption import caption_json_to_prompt


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class SamplePaths:
    stem: str
    rgb_path: Path
    pcd_path: Path
    prompt_path: Path
    mask_path: Path


@dataclass(frozen=True)
class PreparedSample:
    batch: dict[str, Any]
    stem: str
    prompt: str
    first_frame: np.ndarray
    masked_first_frame: np.ndarray
    spatial_keep_mask: np.ndarray
    temporal_guidance_mask: np.ndarray
    pcd_frames: np.ndarray
    fps: float


def _index_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing input directory: {directory}")
    return {path.name: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes}


def discover_samples(
    data_root: Path,
    *,
    rgb_dir: str = "rgb_videos",
    pcd_dir: str = "pcd_videos",
    prompt_dir: str = "prompts",
    mask_dir: str = "mask_videos",
    pcd_suffix: str = "_pcd",
    prompt_suffix: str = "_prompt",
    mask_suffix: str = "_mask",
) -> list[SamplePaths]:
    """Pair files by stem; fail on any incomplete sample instead of silently dropping it."""
    root = data_root.expanduser().resolve()
    rgb_files = _index_files(root / rgb_dir, VIDEO_SUFFIXES)
    pcd_files = _index_files(root / pcd_dir, VIDEO_SUFFIXES)
    prompt_files = _index_files(root / prompt_dir, {".json", ".txt"})
    mask_root = root / mask_dir
    if not mask_root.is_dir() and mask_dir == "mask_videos":
        mask_root = root / "mask"
    mask_files = _index_files(mask_root, VIDEO_SUFFIXES | {".png", ".jpg", ".jpeg"})

    if not rgb_files:
        raise ValueError(f"No RGB videos found in {root / rgb_dir}")

    samples = []
    missing: list[str] = []
    for rgb_name, rgb_path in sorted(rgb_files.items()):
        stem = rgb_path.stem
        pcd_name = f"{stem}{pcd_suffix}{rgb_path.suffix}"
        prompt_candidates = [f"{stem}{prompt_suffix}.json", f"{stem}{prompt_suffix}.txt", f"{stem}.json"]
        mask_candidates = [
            f"{stem}{mask_suffix}.mp4",
            f"{stem}{mask_suffix}.mov",
            f"{stem}{mask_suffix}.mkv",
            f"{stem}{mask_suffix}.webm",
            f"{stem}{mask_suffix}.png",
            f"{stem}{mask_suffix}.jpg",
            f"{stem}{mask_suffix}.jpeg",
            f"{stem}.mp4",
            f"{stem}.png",
        ]
        prompt_path = next((prompt_files[name] for name in prompt_candidates if name in prompt_files), None)
        mask_path = next((mask_files[name] for name in mask_candidates if name in mask_files), None)
        absent = []
        if pcd_name not in pcd_files:
            absent.append(pcd_name)
        if prompt_path is None:
            absent.append(f"{stem}{prompt_suffix}.json")
        if mask_path is None:
            absent.append(f"{stem}{mask_suffix}.mp4 or image")
        if absent:
            missing.append(f"{rgb_name}: missing {', '.join(absent)}")
            continue
        samples.append(SamplePaths(stem, rgb_path, pcd_files[pcd_name], prompt_path, mask_path))

    if missing:
        preview = "\n".join(missing[:12])
        suffix = f"\n... and {len(missing) - 12} more" if len(missing) > 12 else ""
        raise FileNotFoundError(f"Unpaired inputs under {root}:\n{preview}{suffix}")
    return samples


def load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() != ".json":
        if not text:
            raise ValueError(f"Prompt is empty: {path}")
        return text
    raw = json.loads(text)
    if isinstance(raw, dict):
        for key in ("caption", "prompt", "text"):
            if isinstance(raw.get(key), str) and raw[key].strip():
                return raw[key].strip()
        if "caption_json" in raw:
            raw = raw["caption_json"]
        if isinstance(raw, dict):
            raw = caption_json_to_prompt(raw)
    prompt = str(raw).strip()
    if not prompt:
        raise ValueError(f"Prompt is empty: {path}")
    return prompt


def _read_video_frames(path: Path, *, height: int, width: int, limit: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    decoder = ffmpeg_decode_video(str(path), scale_hw=(height, width), num_threads=2)
    try:
        for frame in decoder:
            frames.append(frame)
            if len(frames) >= limit:
                break
    finally:
        decoder.close()
    if not frames:
        raise ValueError(f"Video has no decodable frames: {path}")
    return np.stack(frames)


def _read_first_frame(path: Path, *, height: int, width: int) -> np.ndarray:
    return _read_video_frames(path, height=height, width=width, limit=1)[0]


def load_first_frame_mask(path: Path, *, height: int, width: int) -> np.ndarray:
    """Read only mask frame 0; white (>=128) means keep and black means generate."""
    if path.suffix.lower() in VIDEO_SUFFIXES:
        decoder = ffmpeg_decode_video(str(path), scale_hw=None, num_threads=1)
        try:
            frame = next(decoder)
        except StopIteration as error:
            raise ValueError(f"Mask video has no frames: {path}") from error
        finally:
            decoder.close()
        gray = np.rint(frame.astype(np.float32).mean(axis=-1)).astype(np.uint8)
    else:
        with Image.open(path) as image:
            image = image.convert("L")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.NEAREST)
            gray = np.asarray(image, dtype=np.uint8)
    if gray.shape != (height, width):
        gray = np.asarray(
            Image.fromarray(gray).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8
        )
    keep = gray >= 128
    if not keep.any():
        raise ValueError(f"Mask frame 0 contains no white keep pixels: {path}")
    return keep


def build_temporal_guidance_mask(spatial_keep_mask: np.ndarray, num_frames: int = 93) -> np.ndarray:
    """Return [T,H,W] guidance with the spatial keep mask at t=0 and zeros afterward."""
    if spatial_keep_mask.ndim != 2:
        raise ValueError(f"Expected a 2D spatial mask, got {spatial_keep_mask.shape}")
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    temporal = np.zeros((num_frames, *spatial_keep_mask.shape), dtype=bool)
    temporal[0] = spatial_keep_mask.astype(bool, copy=False)
    return temporal


def mask_first_frame(first_frame: np.ndarray, keep_mask: np.ndarray, fill_value: int = 0) -> np.ndarray:
    if first_frame.ndim != 3 or first_frame.shape[-1] != 3 or first_frame.shape[:2] != keep_mask.shape:
        raise ValueError(f"RGB frame and mask dimensions differ: {first_frame.shape} vs {keep_mask.shape}")
    result = np.full_like(first_frame, fill_value)
    result[keep_mask] = first_frame[keep_mask]
    return result


def _normalize_video(frames: np.ndarray) -> torch.Tensor:
    """uint8 [T,H,W,3] -> float32 [3,T,H,W] in the Cosmos tokenizer range [-1,1]."""
    return torch.from_numpy(np.ascontiguousarray(frames)).float().permute(3, 0, 1, 2) / 127.5 - 1.0


def build_model_batch(
    *,
    stem: str,
    masked_first_frame: np.ndarray,
    pcd_frames: np.ndarray,
    prompt: str,
    prompt_key: str,
    fps: float,
    device: str | torch.device,
) -> dict[str, Any]:
    """Build [masked RGB image, full PCD video, generated RGB shape placeholder] + text."""
    height, width = masked_first_frame.shape[:2]
    if pcd_frames.ndim != 4 or pcd_frames.shape[1:] != masked_first_frame.shape:
        raise ValueError(f"PCD/RGB dimensions differ: {pcd_frames.shape} vs {masked_first_frame.shape}")
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")

    video_items = [
        _normalize_video(masked_first_frame[None]),
        _normalize_video(pcd_frames),
        _normalize_video(np.zeros_like(pcd_frames)),
    ]
    image_size = torch.tensor([[height, width, height, width]], dtype=torch.float32, device=device)
    return {
        "__key__": stem,
        "dataset_name": "cosmos3_mask_video_guidance_inference",
        "video": [item.unsqueeze(0).to(device=device) for item in video_items],
        "image_size": [image_size, image_size.clone(), image_size.clone()],
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
    sample: SamplePaths,
    *,
    prompt_key: str,
    num_frames: int,
    temporal_compression: int,
    width: int,
    height: int,
    mask_fill_value: int,
    fps_override: float | None,
    device: str | torch.device,
) -> PreparedSample:
    if num_frames != 1 + (num_frames - 1) // temporal_compression * temporal_compression:
        raise ValueError(
            f"num_frames={num_frames} must be 1 + k*{temporal_compression} for the Cosmos3 video VAE"
        )
    first_frame = _read_first_frame(sample.rgb_path, height=height, width=width)
    keep_mask = load_first_frame_mask(sample.mask_path, height=height, width=width)
    temporal_mask = build_temporal_guidance_mask(keep_mask, num_frames)
    masked_first = mask_first_frame(first_frame, keep_mask, mask_fill_value)

    pcd_info = get_video_metadata(str(sample.pcd_path))
    if pcd_info["total_frames"] and pcd_info["total_frames"] < num_frames:
        raise ValueError(f"PCD has {pcd_info['total_frames']} frames; {num_frames} required: {sample.pcd_path}")
    pcd_frames = _read_video_frames(sample.pcd_path, height=height, width=width, limit=num_frames)
    if len(pcd_frames) != num_frames:
        raise ValueError(f"Decoded {len(pcd_frames)} PCD frames; {num_frames} required: {sample.pcd_path}")
    prompt = load_prompt(sample.prompt_path)
    fps = float(pcd_info["fps"] if fps_override is None else fps_override)
    batch = build_model_batch(
        stem=sample.stem,
        masked_first_frame=masked_first,
        pcd_frames=pcd_frames,
        prompt=prompt,
        prompt_key=prompt_key,
        fps=fps,
        device=device,
    )
    return PreparedSample(
        batch=batch,
        stem=sample.stem,
        prompt=prompt,
        first_frame=first_frame,
        masked_first_frame=masked_first,
        spatial_keep_mask=keep_mask,
        temporal_guidance_mask=temporal_mask,
        pcd_frames=pcd_frames,
        fps=fps,
    )
