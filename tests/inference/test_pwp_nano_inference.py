import numpy as np
import pytest
import torch

from cosmos_framework.data.generator.sequence_packing.packers import is_item_generated
from cosmos_framework.inference.cosmos3_no_video_path_blended_decode import (
    decode_erp_blended,
    erode_keep_mask,
    hardlock_first_frame,
)
from cosmos_framework.inference.pwp_nano_inference import (
    build_pwp_nano_inference_batch,
    build_temporal_guidance_mask,
)


def test_temporal_guidance_mask_keeps_only_spatial_mask_at_frame_zero():
    keep = np.zeros((3, 5), dtype=bool)
    keep[:, :2] = True

    mask = build_temporal_guidance_mask(keep, num_frames=4)

    assert mask.dtype == np.bool_
    assert mask.shape == (4, 3, 5)
    assert np.array_equal(mask[0], keep)
    assert not mask[1:].any()


def test_build_pwp_nano_inference_batch_uses_three_item_contract():
    first = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    pcd = np.full((5, 4, 6, 3), 200, dtype=np.uint8)
    keep = np.zeros((4, 6), dtype=bool)
    keep[:, :3] = True

    batch = build_pwp_nano_inference_batch(
        first_frame=first,
        pcd_frames=pcd,
        keep_mask=keep,
        prompt="prompt",
        prompt_key="ai_caption",
        fps=24,
        device="cpu",
    )

    assert batch["num_vision_items_per_sample"] == [3]
    assert len(batch["video"]) == 3
    assert [tuple(item.shape) for item in batch["video"]] == [
        (1, 3, 1, 4, 6),
        (1, 3, 5, 4, 6),
        (1, 3, 5, 4, 6),
    ]
    assert torch.all(batch["video"][0][..., :, 3:] == -1)
    assert torch.allclose(batch["video"][1], torch.full_like(batch["video"][1], 200 / 127.5 - 1))
    assert torch.all(batch["video"][2] == -1)
    assert batch["sequence_plan"][0].condition_frame_indexes_vision == []
    assert not is_item_generated([], item_idx=0, num_items=3, latent_t=1)
    assert not is_item_generated([], item_idx=1, num_items=3, latent_t=5)
    assert is_item_generated([], item_idx=2, num_items=3, latent_t=5)


def test_build_pwp_nano_inference_batch_keeps_pcd_length_without_rgb_video():
    first = np.full((2, 3, 3), 255, dtype=np.uint8)
    pcd = np.full((9, 2, 3, 3), 127, dtype=np.uint8)
    keep = np.ones((2, 3), dtype=bool)

    batch = build_pwp_nano_inference_batch(
        first_frame=first,
        pcd_frames=pcd,
        keep_mask=keep,
        prompt="prompt",
        prompt_key="ai_caption",
        fps=24,
        device="cpu",
    )

    assert tuple(batch["video"][0].shape) == (1, 3, 1, 2, 3)
    assert tuple(batch["video"][1].shape) == (1, 3, 9, 2, 3)
    assert tuple(batch["video"][2].shape) == (1, 3, 9, 2, 3)


def test_input_builder_can_keep_full_pcd_while_masking_first_rgb():
    first = np.full((2, 4, 3), 255, dtype=np.uint8)
    pcd = np.full((3, 2, 4, 3), 200, dtype=np.uint8)
    keep = np.zeros((2, 4), dtype=bool)
    keep[:, :2] = True

    batch = build_pwp_nano_inference_batch(
        first_frame=first,
        pcd_frames=pcd,
        keep_mask=keep,
        prompt="prompt",
        prompt_key="ai_caption",
        fps=24,
        device="cpu",
    )

    assert torch.all(batch["video"][0][..., :, 2:] == -1)
    assert torch.all(batch["video"][1] > 0)
    assert torch.all(batch["video"][1] == batch["video"][1][..., :1, :1])


def test_hardlock_first_frame_restores_white_region_only():
    decoded = torch.zeros(1, 3, 2, 2, 4)
    first = np.full((2, 4, 3), 255, dtype=np.uint8)
    keep = np.zeros((2, 4), dtype=bool)
    keep[:, :2] = True

    output = hardlock_first_frame(decoded, rgb_first_frame=first, keep_mask=keep)

    assert torch.all(output[:, :, 0, :, :2] == 1)
    assert torch.all(output[:, :, 0, :, 2:] == 0)
    assert torch.all(output[:, :, 1] == 0)


def test_hardlock_erodes_only_the_restored_region():
    decoded = torch.zeros(1, 3, 2, 11, 11)
    first = np.full((11, 11, 3), 255, dtype=np.uint8)
    keep = np.zeros((11, 11), dtype=bool)
    keep[2:9, 2:9] = True

    eroded = erode_keep_mask(keep, 2)
    output = hardlock_first_frame(decoded, rgb_first_frame=first, keep_mask=keep, erode_px=2)

    assert eroded.sum() == 9
    assert torch.all(output[:, :, 0, 4:7, 4:7] == 1)
    assert torch.all(output[:, :, 0, 2:4, 2:9] == 0)
    assert torch.all(output[:, :, 1] == 0)


def test_decode_erp_blended_runs_both_rotations_and_preserves_shape():
    calls = []

    def decoder(latent):
        calls.append(latent.clone())
        return latent.expand(-1, 3, -1, -1, -1) + len(calls)

    latent = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 1, 8)
    decoded = decode_erp_blended(decoder, latent)

    assert len(calls) == 2
    assert torch.equal(calls[1], torch.roll(latent, shifts=4, dims=-1))
    assert tuple(decoded.shape) == (1, 3, 1, 1, 8)


def test_decode_erp_blended_rejects_odd_latent_width():
    with pytest.raises(ValueError, match="even latent width"):
        decode_erp_blended(lambda value: value, torch.zeros(1, 1, 1, 2, 5))
