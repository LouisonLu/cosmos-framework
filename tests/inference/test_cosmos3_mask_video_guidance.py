# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import torch

from cosmos_framework.inference.cosmos3_mask_video_guidance.decode import composite_first_frame
from cosmos_framework.inference.cosmos3_mask_video_guidance.inputs import (
    build_temporal_guidance_mask,
    erode_keep_mask,
    mask_first_frame,
)
from cosmos_framework.inference.cosmos3_mask_video_guidance.sampler import (
    Frame0MaskGuidanceSampler,
    PackedVisionLayout,
    latent_keep_weights,
)


def test_temporal_mask_has_spatial_keep_only_on_first_frame():
    spatial = np.zeros((4, 7), dtype=bool)
    spatial[:, :3] = True
    temporal = build_temporal_guidance_mask(spatial, 93)

    assert temporal.shape == (93, 4, 7)
    np.testing.assert_array_equal(temporal[0], spatial)
    assert not temporal[1:].any()


def test_masked_first_rgb_zeros_only_generation_region():
    rgb = np.full((3, 5, 3), 200, dtype=np.uint8)
    keep = np.zeros((3, 5), dtype=bool)
    keep[:, :2] = True

    result = mask_first_frame(rgb, keep)

    assert np.all(result[:, :2] == 200)
    assert np.all(result[:, 2:] == 0)


def test_erode_keep_mask_shrinks_white_region_by_requested_pixels():
    keep = np.ones((48, 60), dtype=bool)

    eroded = erode_keep_mask(keep, 16)

    assert not eroded[:16].any()
    assert not eroded[-16:].any()
    assert not eroded[:, :16].any()
    assert not eroded[:, -16:].any()
    assert eroded[16:-16, 16:-16].all()


def test_erode_keep_mask_zero_preserves_source_mask():
    keep = np.zeros((10, 12), dtype=bool)
    keep[2:8, 3:9] = True

    eroded = erode_keep_mask(keep, 0)

    np.testing.assert_array_equal(eroded, keep)
    assert eroded is not keep


def test_latent_mask_rejects_nonblack_future_frames():
    temporal = np.zeros((5, 4, 4), dtype=bool)
    temporal[0, :, :2] = True
    temporal[2, 0, 0] = True

    try:
        latent_keep_weights(temporal, (2, 2))
    except ValueError as error:
        assert "after frame 0" in str(error)
    else:
        raise AssertionError("Expected nonblack future mask frame to fail")


def test_sampler_guides_target_frame_zero_only():
    layout = PackedVisionLayout(channels=1, height=2, width=2, item_frames=(1, 2, 2))
    mask = np.zeros((5, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    sampler = Frame0MaskGuidanceSampler(object(), layout=layout, temporal_mask=mask)
    state = torch.arange(layout.total_numel, dtype=torch.float32)
    velocity = torch.zeros_like(state)
    reference = torch.tensor([[[-10.0, -10.0], [-10.0, -10.0]]])

    guided = sampler.lock_velocity(velocity, state, reference, sigma=2.0)

    assert guided[layout.item_offset(0) : layout.item_offset(1)].eq(0).all()
    target_offset = layout.item_offset(2)
    assert guided[target_offset].item() == (state[target_offset].item() - -10.0) / 2.0
    assert guided[target_offset + 1 :].eq(0).all()


def test_pixel_composite_restores_white_keep_area_on_frame_zero():
    video = torch.zeros((1, 3, 3, 2, 4), dtype=torch.float32)
    source = np.full((2, 4, 3), 255, dtype=np.uint8)
    keep = np.zeros((2, 4), dtype=bool)
    keep[:, :2] = True

    output = composite_first_frame(video, source, keep)

    assert torch.all(output[0, :, 0, :, :2] == 1)
    assert torch.all(output[0, :, 0, :, 2:] == 0)
    assert torch.all(output[0, :, 1:] == 0)
