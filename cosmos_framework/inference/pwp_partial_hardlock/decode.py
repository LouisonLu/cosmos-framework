# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""ERP blended VAE decode and optional exact frame-0 pixel compositing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def decode_erp_blended(decoder: Callable[[torch.Tensor], torch.Tensor], latent: torch.Tensor) -> torch.Tensor:
    """Decode an ERP latent at 0 and 180 degrees and blend the two across the left/right seam.

    The rolled decode sees the original seam in the middle of its input; after rolling it
    back it carries most weight near the ERP border, the unrolled decode near the center.
    """
    if latent.ndim != 5:
        raise ValueError(f"Expected latent [B,C,T,H,W], got {tuple(latent.shape)}")
    if latent.shape[-1] < 2 or latent.shape[-1] % 2:
        raise ValueError(f"Blended ERP decoding requires an even latent width, got {latent.shape[-1]}")

    primary = decoder(latent)
    if primary.ndim != 5 or primary.shape[-1] % 2:
        raise ValueError(f"Blended ERP decoding requires an even-width [B,C,T,H,W] decode, got {tuple(primary.shape)}")
    rotated = decoder(torch.roll(latent, shifts=latent.shape[-1] // 2, dims=-1))
    if rotated.shape != primary.shape:
        raise ValueError(f"Rotated decode shape {tuple(rotated.shape)} does not match {tuple(primary.shape)}")
    rotated = torch.roll(rotated, shifts=-(rotated.shape[-1] // 2), dims=-1)

    x = torch.linspace(0, 1, primary.shape[-1], device=primary.device, dtype=primary.dtype)
    primary_weight = (1 - torch.abs(2 * x - 1)).view(1, 1, 1, 1, -1)
    return primary * primary_weight + rotated * (1 - primary_weight)


def composite_first_frame(video: torch.Tensor, first_frame: np.ndarray, keep_mask: np.ndarray) -> torch.Tensor:
    """Replace the ``keep_mask`` region of frame 0 of a ``[1,3,T,H,W]`` video in [0,1] with source pixels."""
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError(f"Expected video [1,3,T,H,W], got {tuple(video.shape)}")
    if first_frame.shape != (*keep_mask.shape, 3) or keep_mask.shape != tuple(video.shape[-2:]):
        raise ValueError("first_frame, keep_mask, and video spatial shapes do not match")
    reference = torch.from_numpy(np.array(first_frame, copy=True)).permute(2, 0, 1).to(video) / 255.0
    mask = torch.from_numpy(np.array(keep_mask, copy=True)).to(video)
    output = video.clone()
    output[0, :, 0] = mask * reference + (1.0 - mask) * output[0, :, 0]
    return output
