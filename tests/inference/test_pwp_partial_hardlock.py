# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_framework.inference.pwp_partial_hardlock.config import PartialHardlockRequest, load_requests
from cosmos_framework.inference.pwp_partial_hardlock.decode import composite_first_frame, decode_erp_blended
from cosmos_framework.inference.pwp_partial_hardlock.latent_hardlock import (
    PartialHardlockSampler,
    VisionItemLayout,
    erode_keep_mask,
    latent_weight_map,
)
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

LAYOUT = VisionItemLayout(channels=2, height=2, width=4, item_frames=(1, 3, 3))


def _toy_velocity_fn(model_x0: torch.Tensor):
    """Flow velocity toward ``model_x0`` with zero velocity on the two clean condition items."""
    generated_start = LAYOUT.offset(2)

    def velocity_fn(noise_x: list[torch.Tensor], timestep: torch.Tensor) -> list[torch.Tensor]:
        sigma = float(timestep.reshape(-1)[0]) / 1000.0
        outputs = []
        for state in noise_x:
            velocity = torch.zeros_like(state)
            generated = state[generated_start:]
            velocity[generated_start:] = (generated - model_x0) / max(sigma, 1e-6)
            outputs.append(velocity)
        return outputs

    return velocity_fn


def _initial_state(reference: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    state = torch.randn(LAYOUT.total, generator=generator)
    LAYOUT.frame0(state, 0).copy_(reference)  # clean condition item, as _prepare_inference_data builds it
    return state


@pytest.mark.L0
def test_lock_velocity_sets_x0_prediction_to_reference_inside_weight():
    weight = torch.zeros(2, 4)
    weight[:, :2] = 1.0
    sampler = PartialHardlockSampler(lambda *a, **k: None, layout=LAYOUT, weight=weight)
    state = torch.randn(LAYOUT.total)
    velocity = torch.randn(LAYOUT.total)
    reference = torch.randn(2, 2, 4)
    sigma = 0.37

    locked = sampler.lock_velocity(velocity, state, reference, sigma)

    x0_prediction = LAYOUT.frame0(state, 2) - sigma * LAYOUT.frame0(locked, 2)
    assert torch.allclose(x0_prediction[..., :2], reference[..., :2], atol=1e-5)
    assert torch.equal(LAYOUT.frame0(locked, 2)[..., 2:], LAYOUT.frame0(velocity, 2)[..., 2:])
    assert torch.equal(locked[: LAYOUT.offset(2)], velocity[: LAYOUT.offset(2)])
    target = slice(LAYOUT.offset(2), LAYOUT.total)
    assert torch.equal(locked[target].view(2, 3, 2, 4)[:, 1:], velocity[target].view(2, 3, 2, 4)[:, 1:])


@pytest.mark.L0
def test_unipc_partial_hardlock_converges_to_reference_and_leaves_rest_free():
    reference = torch.randn(2, 2, 4, generator=torch.Generator().manual_seed(1))
    model_x0 = torch.randn(LAYOUT.numel(2), generator=torch.Generator().manual_seed(2))
    weight = torch.zeros(2, 4)
    weight[:, 1:3] = 1.0
    state = _initial_state(reference)
    base = UniPCSampler(tensor_kwargs={"device": "cpu"})

    locked = PartialHardlockSampler(base, layout=LAYOUT, weight=weight)(
        _toy_velocity_fn(model_x0), [state.clone()], num_steps=12, shift=5.0, seed=[3]
    )[0]
    free = base(_toy_velocity_fn(model_x0), [state.clone()], num_steps=12, shift=5.0, seed=[3])[0]

    locked_frame0 = LAYOUT.frame0(locked, 2)
    assert torch.allclose(locked_frame0[..., 1:3], reference[..., 1:3], atol=1e-4)
    assert torch.allclose(locked_frame0[..., [0, 3]], LAYOUT.frame0(free, 2)[..., [0, 3]], atol=1e-5)
    target = slice(LAYOUT.offset(2), LAYOUT.total)
    assert torch.allclose(locked[target].view(2, 3, 2, 4)[:, 1:], free[target].view(2, 3, 2, 4)[:, 1:], atol=1e-5)
    assert torch.allclose(locked[: LAYOUT.offset(2)], state[: LAYOUT.offset(2)], atol=1e-5)


@pytest.mark.L0
def test_sampler_rejects_a_noisy_reference_item():
    def velocity_fn(noise_x, _timestep):
        return [torch.ones_like(state) for state in noise_x]

    sampler = PartialHardlockSampler(
        UniPCSampler(tensor_kwargs={"device": "cpu"}),
        layout=LAYOUT,
        weight=torch.ones(2, 4),
    )
    with pytest.raises(RuntimeError, match="not a clean condition item"):
        sampler(velocity_fn, [torch.zeros(LAYOUT.total)], num_steps=2, shift=1.0, seed=[0])


@pytest.mark.L0
def test_sampler_rejects_a_base_sampler_with_a_different_schedule():
    base = UniPCSampler(tensor_kwargs={"device": "cpu"})
    sampler = PartialHardlockSampler(base, layout=LAYOUT, weight=torch.ones(2, 4))

    def mismatched_base(velocity_fn, noise, **kwargs):
        return base(velocity_fn, noise, **{**kwargs, "num_steps": kwargs["num_steps"] + 1})

    sampler.base_sampler = mismatched_base
    mismatched_base.cfg = base.cfg
    with pytest.raises(RuntimeError, match="does not match the UniPC schedule"):
        sampler(
            _toy_velocity_fn(torch.zeros(LAYOUT.numel(2))),
            [_initial_state(torch.zeros(2, 2, 4))],
            num_steps=4,
            shift=5.0,
            seed=[0],
        )


@pytest.mark.L0
def test_sampler_rejects_a_state_with_extra_modalities():
    sampler = PartialHardlockSampler(lambda *a, **k: None, layout=LAYOUT, weight=torch.ones(2, 4))
    with pytest.raises(ValueError, match="expects"):
        sampler(lambda *a: None, [torch.zeros(LAYOUT.total + 5)], num_steps=1, shift=1.0, seed=[0])


@pytest.mark.L0
def test_erode_and_latent_weight_map_follow_cosmos25_rules():
    keep = np.zeros((32, 64), dtype=bool)
    keep[:, :32] = True

    eroded = erode_keep_mask(keep, 4)
    weight = latent_weight_map(eroded, (2, 4))

    assert eroded[:, :28].all() and not eroded[:, 28:].any()
    assert torch.equal(latent_weight_map(np.ones((32, 64), dtype=bool), (2, 4)), torch.ones(2, 4))
    assert weight.shape == (2, 4) and torch.all(weight[:, :1] == 1) and torch.all(weight[:, 2:] == 0)


@pytest.mark.L0
def test_decode_erp_blended_rolls_and_blends():
    calls = []

    def decoder(latent):
        calls.append(latent.clone())
        return latent.repeat_interleave(2, dim=-1)

    latent = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 1, 8)
    decoded = decode_erp_blended(decoder, latent)

    assert len(calls) == 2 and torch.equal(calls[1], torch.roll(latent, 4, dims=-1))
    assert decoded.shape == (1, 1, 1, 1, 16)
    assert torch.allclose(decoded, latent.repeat_interleave(2, dim=-1))
    with pytest.raises(ValueError, match="even latent width"):
        decode_erp_blended(decoder, torch.zeros(1, 1, 1, 1, 3))


@pytest.mark.L0
def test_composite_first_frame_only_replaces_frame0_keep_region():
    video = torch.zeros(1, 3, 2, 2, 2)
    first = np.full((2, 2, 3), 255, dtype=np.uint8)
    keep = np.array([[True, False], [False, False]])

    output = composite_first_frame(video, first, keep)

    assert torch.all(output[0, :, 0, 0, 0] == 1) and output[0, :, 0].sum() == 3 and output[0, :, 1].sum() == 0


@pytest.mark.L0
def test_load_requests_resolves_relative_paths_and_rejects_unknown_fields(tmp_path: Path):
    request = {"name": "a", "prompt": "a road", "image_context_path": "f0.png", "pcd_path": "p.mp4"}
    (tmp_path / "req.jsonl").write_text(json.dumps(request) + "\n" + json.dumps({**request, "name": "b"}) + "\n")

    loaded = load_requests(tmp_path / "req.jsonl")

    assert [item.name for item in loaded] == ["a", "b"]
    assert loaded[0].image_context_path == (tmp_path / "f0.png").resolve()
    assert loaded[0].blended_decoding and loaded[0].guided_generation_mask_erode_px == 8
    with pytest.raises(ValueError):
        PartialHardlockRequest.model_validate({**request, "video_path": "rgb.mp4"})
