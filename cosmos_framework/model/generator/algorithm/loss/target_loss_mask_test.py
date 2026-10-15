# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.algorithm.loss.flow_matching import compute_flow_matching_loss
from cosmos_framework.model.generator.algorithm.loss.target_loss_mask import build_latent_valid_mask


class _UnitWeightFlow:
    def train_time_weight(self, timesteps, tensor_kwargs):
        return torch.ones_like(timesteps, **tensor_kwargs)


@pytest.mark.L0
def test_latent_mask_excludes_black_source_bins() -> None:
    pixel_mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])
    latent_mask = build_latent_valid_mask(pixel_mask, (1, 1, 2))
    torch.testing.assert_close(latent_mask, torch.tensor([[[[1.0, 0.0]]]]))


@pytest.mark.L0
def test_target_loss_mask_excludes_black_regions_from_numerator_and_denominator() -> None:
    pred = [torch.tensor([[[[1.0, 100.0]]]])]
    target = [torch.zeros_like(pred[0])]
    condition_mask = [torch.zeros(1, 1, 1)]
    target_loss_mask = [torch.tensor([[[[1.0, 0.0]]]])]

    weighted, per_instance = compute_flow_matching_loss(
        pred=pred,
        target=target,
        condition_mask=condition_mask,
        timesteps=torch.zeros(1, 1),
        has_valid_tokens=True,
        rectified_flow=_UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        target_loss_mask=target_loss_mask,
    )

    # The black pixel's 100^2 error is excluded, and the denominator counts
    # only the one valid target element, so the loss is exactly 1.
    torch.testing.assert_close(weighted, torch.tensor(1.0))
    torch.testing.assert_close(per_instance, torch.tensor([1.0]))


@pytest.mark.L0
def test_target_loss_mask_supports_batched_vision_latents() -> None:
    # Cosmos3 keeps a singleton batch dimension on each vision latent:
    # [B,C,T,H,W]. Both channels contribute to the valid first cell, while the
    # black cell must be excluded from the numerator and denominator.
    pred = [torch.tensor([[[[[1.0, 100.0]]], [[[2.0, 200.0]]]]])]
    target = [torch.zeros_like(pred[0])]
    condition_mask = [torch.zeros(1, 1, 1)]
    target_loss_mask = [torch.tensor([[[[1.0, 0.0]]]])]

    weighted, per_instance = compute_flow_matching_loss(
        pred=pred,
        target=target,
        condition_mask=condition_mask,
        timesteps=torch.zeros(1, 1),
        has_valid_tokens=True,
        rectified_flow=_UnitWeightFlow(),
        tensor_kwargs_fp32={"dtype": torch.float32},
        target_loss_mask=target_loss_mask,
    )

    # (1^2 + 2^2) / 2 valid channel elements.
    torch.testing.assert_close(weighted, torch.tensor(2.5))
    torch.testing.assert_close(per_instance, torch.tensor([2.5]))
