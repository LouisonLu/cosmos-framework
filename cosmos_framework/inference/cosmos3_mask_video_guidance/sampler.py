# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 UniPC velocity guidance for frame-0 spatial keep masks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler


@dataclass(frozen=True)
class PackedVisionLayout:
    """Flattened [C,T,H,W] vision-item offsets used by Cosmos3's flow sampler."""

    channels: int
    height: int
    width: int
    item_frames: tuple[int, ...]
    temporal_compression_factor: int = 4

    def item_numel(self, index: int) -> int:
        return self.channels * self.item_frames[index] * self.height * self.width

    def item_offset(self, index: int) -> int:
        return sum(self.item_numel(i) for i in range(index))

    @property
    def total_numel(self) -> int:
        return sum(self.item_numel(i) for i in range(len(self.item_frames)))

    def frame(self, flat: torch.Tensor, item_index: int, frame_index: int = 0) -> torch.Tensor:
        item_index %= len(self.item_frames)
        if frame_index < 0 or frame_index >= self.item_frames[item_index]:
            raise IndexError(
                f"frame {frame_index} outside item {item_index} "
                f"with {self.item_frames[item_index]} frames"
            )
        start = self.item_offset(item_index)
        item = flat[start : start + self.item_numel(item_index)].view(
            self.channels, self.item_frames[item_index], self.height, self.width
        )
        return item[:, frame_index]


def latent_keep_weights(temporal_mask: np.ndarray, latent_hw: tuple[int, int]) -> torch.Tensor:
    """Resize only temporal mask frame 0 to latent space; later mask frames must be black."""
    if temporal_mask.ndim != 3 or temporal_mask.shape[0] < 1:
        raise ValueError(f"Expected temporal mask [T,H,W], got {temporal_mask.shape}")
    if np.any(temporal_mask[1:]):
        raise ValueError("Temporal mask guidance must be all black after frame 0")
    spatial = torch.from_numpy(np.ascontiguousarray(temporal_mask[0], dtype=np.float32))[None, None]
    return F.interpolate(spatial, size=latent_hw, mode="bilinear", align_corners=False)[0, 0].clamp_(0, 1)


class Frame0MaskGuidanceSampler:
    """Wrap Cosmos3 UniPC and steer only target video frame 0 toward its clean reference.

    The temporal mask is sampler guidance, not a fourth model vision item. Under
    Cosmos3's flow convention x0 = x_t - sigma*v, replacing target-frame-0
    velocity by (x_t - x0_reference)/sigma gives the locked cells the reference
    x0 prediction at each denoising step.
    """

    def __init__(
        self,
        base_sampler: Callable[..., Any],
        *,
        layout: PackedVisionLayout,
        temporal_mask: np.ndarray,
        reference_item: int = 0,
        target_item: int = 2,
        min_sigma: float = 1e-6,
    ) -> None:
        if layout.item_frames[reference_item] != 1:
            raise ValueError("Reference item must contain one latent frame")
        if layout.item_frames[target_item] <= 1:
            raise ValueError("Generated target item must be a video")
        expected_frames = 1 + (layout.item_frames[target_item] - 1) * layout.temporal_compression_factor
        if temporal_mask.shape[0] != expected_frames:
            raise ValueError(
                "Temporal mask frame count does not match Cosmos3 target latent length: "
                f"mask={temporal_mask.shape[0]}, target_latents={layout.item_frames[target_item]}, "
                f"temporal_compression={layout.temporal_compression_factor}"
            )
        self.base_sampler = base_sampler
        self.layout = layout
        self.weight = latent_keep_weights(temporal_mask, (layout.height, layout.width))
        self.reference_item = reference_item
        self.target_item = target_item % len(layout.item_frames)
        self.min_sigma = min_sigma

    def _check_flat(self, tensor: torch.Tensor, label: str) -> None:
        if tensor.ndim != 1 or tensor.numel() != self.layout.total_numel:
            raise ValueError(
                f"{label} has shape {tuple(tensor.shape)}; expected a flat Cosmos3 PwP state "
                f"with {self.layout.total_numel} values ({self.layout})"
            )

    def lock_velocity(
        self,
        velocity: torch.Tensor,
        state: torch.Tensor,
        reference: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        self._check_flat(velocity, "velocity")
        self._check_flat(state, "sampler state")
        sigma = max(float(sigma), self.min_sigma)
        target_state = self.layout.frame(state, self.target_item, 0).float()
        target_velocity = self.layout.frame(velocity, self.target_item, 0)
        weight = self.weight.to(device=velocity.device, dtype=torch.float32)
        locked_velocity = (target_state - reference.to(device=velocity.device, dtype=torch.float32)) / sigma
        output = velocity.clone()
        output_target = self.layout.frame(output, self.target_item, 0)
        output_target.copy_((weight * locked_velocity + (1.0 - weight) * target_velocity.float()).to(output.dtype))
        return output

    def _schedule(self, num_steps: int, shift: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.base_sampler.cfg
        schedule = FlowUniPCMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            shift=cfg.shift,
            use_dynamic_shifting=cfg.use_dynamic_shifting,
        )
        schedule.set_timesteps(num_steps, device="cpu", shift=cfg.shift if shift is None else shift)
        return schedule.sigmas, schedule.timesteps

    def __call__(
        self,
        velocity_fn: Callable[[list[torch.Tensor], torch.Tensor], list[torch.Tensor]],
        noise: list[torch.Tensor],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        if not isinstance(noise, list):
            raise TypeError("Frame0MaskGuidanceSampler expects Cosmos3's per-sample flattened noise list")
        for state in noise:
            self._check_flat(state, "initial sampler state")
        sigmas, timesteps = self._schedule(kwargs["num_steps"], kwargs.get("shift"))
        references = [self.layout.frame(state, self.reference_item).float().clone() for state in noise]
        step_index = 0

        def guided_velocity(states: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:
            nonlocal step_index
            if step_index >= len(timesteps) or int(timestep.reshape(-1)[0]) != int(timesteps[step_index]):
                raise RuntimeError(f"Unexpected UniPC timestep at step {step_index}; refusing misaligned mask guidance")
            velocities = velocity_fn(states, timestep)
            if step_index == 0:
                for velocity in velocities:
                    if torch.count_nonzero(self.layout.frame(velocity, self.reference_item)).item():
                        raise RuntimeError("Reference item is not a clean Cosmos3 condition item")
            current_sigma = float(sigmas[step_index])
            step_index += 1
            return [
                self.lock_velocity(velocity, state, reference, current_sigma)
                for velocity, state, reference in zip(velocities, states, references, strict=True)
            ]

        result = self.base_sampler(guided_velocity, noise, **kwargs)
        if step_index != len(timesteps):
            raise RuntimeError(f"UniPC made {step_index} calls for a {len(timesteps)}-step schedule")
        return result
