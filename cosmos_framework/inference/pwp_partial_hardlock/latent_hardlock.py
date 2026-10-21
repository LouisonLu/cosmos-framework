# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Latent-space partial hardlock of the generated vision item's first frame.

Cosmos2.5 multiplies its conditional-frame mask by a spatial mask, so only the
white region of frame 0 is held at the clean latent. Cosmos3 condition masks are
per latent frame, so this module reaches the same result inside the sampler:

* the reference is the clean latent of the masked first-frame vision item;
* at every step, inside the locked latent cells of the generated item's frame 0,
  the velocity is replaced by ``(x_t - x0_ref) / sigma_t``. With the flow
  convention ``x_t = (1 - sigma) x0 + sigma eps`` (``x0 = x_t - sigma v``), the
  sampler's x0 prediction there is exactly ``x0_ref``, so the state follows the
  noised reference path and reaches ``x0_ref`` at the end;
* the network always sees locked cells at the noise level of the current step,
  never a clean latent tagged with a noisy timestep.

A soft latent weight (from the pixel mask resized like Cosmos2.5) blends the
locked and model velocities near the mask boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler


def erode_keep_mask(keep_mask: np.ndarray, erode_px: int) -> np.ndarray:
    """Shrink every white region by ``erode_px`` pixels (Cosmos2.5 max-pool rule)."""
    if keep_mask.ndim != 2:
        raise ValueError(f"Expected a 2D keep mask, got {keep_mask.shape}")
    if erode_px < 0:
        raise ValueError(f"erode_px must be non-negative, got {erode_px}")
    if erode_px == 0:
        return keep_mask.astype(bool, copy=True)
    invalid = 1.0 - torch.from_numpy(np.ascontiguousarray(keep_mask, dtype=np.float32))[None, None]
    eroded = 1.0 - F.max_pool2d(invalid, kernel_size=2 * erode_px + 1, stride=1, padding=erode_px)
    return eroded[0, 0].numpy() >= 0.5


def latent_weight_map(keep_mask: np.ndarray, latent_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a pixel keep mask to the latent grid as Cosmos2.5 ``construct_latent_weight_map`` does."""
    if keep_mask.ndim != 2:
        raise ValueError(f"Expected a 2D keep mask, got {keep_mask.shape}")
    mask = torch.from_numpy(np.ascontiguousarray(keep_mask, dtype=np.float32))[None, None]
    return F.interpolate(mask, size=latent_hw, mode="bilinear", align_corners=False)[0, 0]  # [h,w] in [0,1]


@dataclass(frozen=True)
class VisionItemLayout:
    """Flattened ``[C,T,H,W]`` layout of one sample's vision items in the sampler state."""

    channels: int
    height: int
    width: int
    item_frames: tuple[int, ...]

    def numel(self, index: int) -> int:
        return self.channels * self.item_frames[index] * self.height * self.width

    def offset(self, index: int) -> int:
        return sum(self.numel(i) for i in range(index))

    @property
    def total(self) -> int:
        return self.offset(len(self.item_frames))

    def frame0(self, flat: torch.Tensor, index: int) -> torch.Tensor:
        """View ``[C,H,W]`` of latent frame 0 of item ``index`` inside a flat state tensor."""
        start = self.offset(index)
        item = flat[start : start + self.numel(index)].view(
            self.channels, self.item_frames[index], self.height, self.width
        )
        return item[:, 0]


class PartialHardlockSampler:
    """Wrap a UniPC sampler so frame 0 of the generated item is locked to a clean reference latent.

    Pass an instance as ``sampler=`` to ``OmniMoTModel.generate_samples_from_batch``;
    it delegates the actual stepping to ``base_sampler`` (a ``UniPCSampler``) and only
    edits the velocity of the locked cells after classifier-free guidance.
    """

    def __init__(
        self,
        base_sampler: Callable[..., Any],
        *,
        layout: VisionItemLayout,
        weight: torch.Tensor,
        reference_item: int = 0,
        target_item: int = -1,
        min_sigma: float = 1e-6,
    ) -> None:
        if weight.shape != (layout.height, layout.width):
            raise ValueError(f"Latent weight {tuple(weight.shape)} does not match {(layout.height, layout.width)}")
        if layout.item_frames[reference_item] != 1:
            raise ValueError("The reference vision item must be a single latent frame")
        self.base_sampler = base_sampler
        self.layout = layout
        self.weight = weight.float()
        self.reference_item = reference_item
        self.target_item = target_item % len(layout.item_frames)
        self.min_sigma = min_sigma

    def _check_state(self, flat: torch.Tensor, label: str) -> None:
        if flat.ndim != 1 or flat.numel() != self.layout.total:
            raise ValueError(
                f"{label} has {flat.numel()} elements; the vision layout {self.layout} expects {self.layout.total}. "
                "Partial hardlock supports exactly the PwP vision items and no other modality."
            )

    def lock_velocity(
        self,
        velocity: torch.Tensor,
        state: torch.Tensor,
        reference: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Return ``velocity`` with the locked cells of the target frame 0 pointed at ``reference``."""
        self._check_state(velocity, "velocity")
        self._check_state(state, "sampler state")
        sigma = max(sigma, self.min_sigma)
        weight = self.weight.to(device=velocity.device)
        target_state = self.layout.frame0(state, self.target_item).float()
        locked = (target_state - reference.to(device=velocity.device)) / sigma

        output = velocity.clone()
        target_velocity = self.layout.frame0(output, self.target_item)
        target_velocity.copy_((weight * locked + (1.0 - weight) * target_velocity.float()).to(output.dtype))
        return output

    def _sigma_schedule(self, num_steps: int, shift: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild the base sampler's schedule: the passed timesteps are int-truncated, the sigmas are not."""
        cfg = self.base_sampler.cfg
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=cfg.num_train_timesteps,
            shift=cfg.shift,
            use_dynamic_shifting=cfg.use_dynamic_shifting,
        )
        scheduler.set_timesteps(num_steps, device="cpu", shift=cfg.shift if shift is None else shift)
        return scheduler.sigmas, scheduler.timesteps

    def __call__(
        self,
        velocity_fn: Callable[[list[torch.Tensor], torch.Tensor], list[torch.Tensor]],
        noise: list[torch.Tensor],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        if not isinstance(noise, list):
            raise TypeError("PartialHardlockSampler expects the flattened per-sample noise list")
        for state in noise:
            self._check_state(state, "initial state")
        sigmas, timesteps = self._sigma_schedule(kwargs["num_steps"], kwargs.get("shift"))
        # Clean condition items enter the sampler state as their exact latent and keep zero velocity.
        references = [self.layout.frame0(state, self.reference_item).float().clone() for state in noise]
        step = 0

        def locked_velocity_fn(noise_x: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:
            nonlocal step
            if step >= len(timesteps) or int(timestep.reshape(-1)[0]) != int(timesteps[step]):
                raise RuntimeError(
                    f"Sampler timestep {timestep.reshape(-1)[0].item()} at step {step} does not match the UniPC "
                    "schedule; partial hardlock needs one velocity call per UniPC step."
                )
            velocities = velocity_fn(noise_x, timestep)
            if step == 0:
                for velocity in velocities:
                    if torch.count_nonzero(self.layout.frame0(velocity, self.reference_item)).item() != 0:
                        raise RuntimeError(
                            "The reference vision item has nonzero velocity, so it is not a clean condition item; "
                            "the batch does not follow the PwP [masked first RGB, PCD, generated RGB] contract."
                        )
            sigma = float(sigmas[step])
            step += 1
            return [
                self.lock_velocity(velocity, state, reference, sigma)
                for velocity, state, reference in zip(velocities, noise_x, references, strict=True)
            ]

        return self.base_sampler(locked_velocity_fn, noise, **kwargs)
