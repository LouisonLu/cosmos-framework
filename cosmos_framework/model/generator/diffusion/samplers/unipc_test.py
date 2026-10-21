# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch

from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler


@pytest.mark.L0
def test_unipc_preserves_conditioned_entries_during_sampling():
    sampler = UniPCSampler(tensor_kwargs={"device": "cpu"})
    noise = [torch.tensor([2.0, 4.0])]
    reference = [torch.tensor([10.0, 8.0])]
    mask = [torch.tensor([1.0, 0.0])]

    result = sampler(
        lambda latent, _time: torch.zeros_like(latent),
        noise,
        num_steps=2,
        shift=1.0,
        seed=[7],
        condition_reference=reference,
        condition_mask=mask,
    )

    assert torch.equal(result[0][0], reference[0][0])
    assert torch.equal(result[0][1], noise[0][1])


@pytest.mark.L0
def test_unipc_conditioning_is_optional_and_reproducible():
    sampler = UniPCSampler(tensor_kwargs={"device": "cpu"})
    noise = [torch.randn(8)]
    velocity = lambda latent, _time: torch.zeros_like(latent)

    result_a = sampler(velocity, noise, num_steps=2, shift=1.0, seed=[17])
    result_b = sampler(velocity, noise, num_steps=2, shift=1.0, seed=[17])

    assert torch.equal(result_a[0], result_b[0])


@pytest.mark.L0
def test_unipc_rejects_mismatched_condition_shapes():
    sampler = UniPCSampler(tensor_kwargs={"device": "cpu"})

    with pytest.raises(ValueError, match="must match the latent shape"):
        sampler(
            lambda latent, _time: torch.zeros_like(latent),
            [torch.zeros(3)],
            num_steps=1,
            shift=1.0,
            seed=[3],
            condition_reference=[torch.zeros(2)],
            condition_mask=[torch.ones(2)],
        )
