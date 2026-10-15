# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pixel-to-latent validity-mask utilities for masked RGB target training."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def build_latent_valid_mask(
    pixel_valid_mask_1_T_H_W: torch.Tensor,
    latent_shape_T_H_W: Sequence[int],
    *,
    spatial_guard_px: int = 0,
    temporal_guard_frames: int = 0,
    min_valid_ratio: float = 0.0,
) -> torch.Tensor:
    """Map a binary pixel mask to a conservative latent validity mask.

    A latent cell is valid only when its complete pooling bin is valid.  This
    makes black pixels contribute exactly zero loss.  Optional guards can
    remove extra boundary cells when VAE receptive-field contamination needs
    to be suppressed.

    Args:
        pixel_valid_mask_1_T_H_W: Float/bool mask shaped ``[1,T,H,W]``.
        latent_shape_T_H_W: Target latent shape ``(T_latent,H_latent,W_latent)``.
        spatial_guard_px: Invalid-pixel dilation in height/width.
        temporal_guard_frames: Invalid-pixel dilation in time.
        min_valid_ratio: Minimum fraction of valid latent cells required for a
            sample. ``0`` disables the coverage check.
    """
    if pixel_valid_mask_1_T_H_W.ndim != 4 or pixel_valid_mask_1_T_H_W.shape[0] != 1:
        raise ValueError(
            "Expected pixel target mask shape [1,T,H,W], "
            f"got {tuple(pixel_valid_mask_1_T_H_W.shape)}"
        )
    if spatial_guard_px < 0 or temporal_guard_frames < 0:
        raise ValueError("Mask guard sizes must be non-negative")
    if not 0.0 <= min_valid_ratio <= 1.0:
        raise ValueError("min_valid_ratio must be in [0, 1]")

    latent_shape = tuple(int(value) for value in latent_shape_T_H_W)
    if len(latent_shape) != 3 or min(latent_shape) <= 0:
        raise ValueError(f"Expected positive latent shape (T,H,W), got {latent_shape}")

    valid = (pixel_valid_mask_1_T_H_W > 0.5).to(dtype=torch.float32).unsqueeze(0)
    invalid = 1.0 - valid
    if spatial_guard_px or temporal_guard_frames:
        kernel = (
            2 * temporal_guard_frames + 1,
            2 * spatial_guard_px + 1,
            2 * spatial_guard_px + 1,
        )
        padding = (temporal_guard_frames, spatial_guard_px, spatial_guard_px)
        invalid = F.max_pool3d(invalid, kernel_size=kernel, stride=1, padding=padding)

    # Max-pool invalidity means a latent cell is valid only if no source pixel
    # in its adaptive pooling bin is black.
    latent_invalid = F.adaptive_max_pool3d(invalid, output_size=latent_shape)
    latent_valid = (latent_invalid < 0.5).to(dtype=torch.float32)
    valid_ratio = latent_valid.mean()
    if valid_ratio < min_valid_ratio:
        raise ValueError(
            "Target loss mask has too little valid latent coverage after guarding: "
            f"ratio={float(valid_ratio):.6f}, minimum={min_valid_ratio:.6f}"
        )
    if not torch.any(latent_valid):
        raise ValueError("Target loss mask contains no valid latent cells")
    return latent_valid.squeeze(0).contiguous()  # [1,T_latent,H_latent,W_latent]


__all__ = ["build_latent_valid_mask"]
