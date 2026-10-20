# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 no-video-path inference decode with ERP blending and frame-0 hardlock."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


def decode_erp_blended(
    decoder: Callable[[torch.Tensor], torch.Tensor],
    latent: torch.Tensor,
) -> torch.Tensor:
    """Decode ERP latents at 0 and 180 degrees and blend across the seam."""
    if latent.ndim != 5:
        raise ValueError(f"Expected latent [B,C,T,H,W], got {tuple(latent.shape)}")
    if latent.shape[-1] < 2 or latent.shape[-1] % 2:
        raise ValueError(f"Blended ERP decoding requires an even latent width, got {latent.shape[-1]}")

    primary = decoder(latent)
    if primary.ndim != 5 or primary.shape[-1] < 2 or primary.shape[-1] % 2:
        raise ValueError(f"Blended ERP decoding requires an even-width [B,C,T,H,W] decode, got {tuple(primary.shape)}")

    rotated_latent = torch.roll(latent, shifts=latent.shape[-1] // 2, dims=-1)
    rotated = decoder(rotated_latent)
    if rotated.shape != primary.shape:
        raise ValueError(f"Rotated decode shape {tuple(rotated.shape)} does not match {tuple(primary.shape)}")
    rotated_aligned = torch.roll(rotated, shifts=-(rotated.shape[-1] // 2), dims=-1)

    x = torch.linspace(0, 1, primary.shape[-1], device=primary.device, dtype=primary.dtype)
    primary_weight = (1 - torch.abs(2 * x - 1)).view(1, 1, 1, 1, -1)
    return primary * primary_weight + rotated_aligned * (1 - primary_weight)


def erode_keep_mask(keep_mask: np.ndarray, erode_px: int) -> np.ndarray:
    """Erode the white keep region using the Cosmos2.5 max-pool rule."""
    if keep_mask.ndim != 2:
        raise ValueError(f"Expected a 2D keep mask, got {keep_mask.shape}")
    if erode_px < 0:
        raise ValueError(f"erode_px must be non-negative, got {erode_px}")
    if erode_px == 0:
        return keep_mask.astype(bool, copy=False)

    binary = torch.from_numpy(np.ascontiguousarray(keep_mask >= 0.5)).float()[None, None]
    invalid = 1.0 - binary
    eroded = 1.0 - F.max_pool2d(
        invalid,
        kernel_size=2 * erode_px + 1,
        stride=1,
        padding=erode_px,
    )
    return eroded[0, 0].numpy() >= 0.5


def hardlock_first_frame(
    decoded_video: torch.Tensor,
    *,
    rgb_first_frame: np.ndarray,
    keep_mask: np.ndarray,
    erode_px: int = 0,
) -> torch.Tensor:
    """Replace the decoded frame-0 keep region with the source RGB pixels."""
    if decoded_video.ndim != 5 or decoded_video.shape[0] != 1:
        raise ValueError(f"Expected decoded video [1,3,T,H,W], got {tuple(decoded_video.shape)}")
    if rgb_first_frame.shape != (*keep_mask.shape, 3):
        raise ValueError("rgb_first_frame and keep_mask shapes do not match")
    keep_mask = erode_keep_mask(keep_mask, erode_px)
    reference = torch.from_numpy(np.ascontiguousarray(rgb_first_frame)).float().permute(2, 0, 1) / 255.0
    reference = reference.to(device=decoded_video.device, dtype=decoded_video.dtype).unsqueeze(0)
    mask = torch.from_numpy(np.ascontiguousarray(keep_mask)).to(device=decoded_video.device)
    mask = mask.to(dtype=decoded_video.dtype).view(1, 1, *keep_mask.shape)
    output = decoded_video.clone()
    output[:, :, 0] = mask * reference + (1.0 - mask) * output[:, :, 0]
    return output
