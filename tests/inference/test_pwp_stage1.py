import numpy as np
import torch

from cosmos_framework.data.generator.sequence_packing.packers import is_item_generated
from cosmos_framework.inference.pwp_stage1 import build_pwp_stage1_batch, hardlock_first_frame


def test_build_pwp_stage1_batch_uses_three_item_contract():
    first = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    pcd = np.full((5, 4, 6, 3), 200, dtype=np.uint8)
    keep = np.zeros((4, 6), dtype=bool)
    keep[:, :3] = True

    batch = build_pwp_stage1_batch(
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
    assert torch.all(batch["video"][1][..., :, 3:] == -1)
    assert torch.all(batch["video"][2] == -1)
    assert batch["sequence_plan"][0].condition_frame_indexes_vision == []
    assert not is_item_generated([], item_idx=0, num_items=3, latent_t=1)
    assert not is_item_generated([], item_idx=1, num_items=3, latent_t=5)
    assert is_item_generated([], item_idx=2, num_items=3, latent_t=5)


def test_build_pwp_stage1_batch_keeps_pcd_length_without_rgb_video():
    first = np.full((2, 3, 3), 255, dtype=np.uint8)
    pcd = np.full((9, 2, 3, 3), 127, dtype=np.uint8)
    keep = np.ones((2, 3), dtype=bool)

    batch = build_pwp_stage1_batch(
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


def test_hardlock_first_frame_restores_white_region_only():
    decoded = torch.zeros(1, 3, 2, 2, 4)
    first = np.full((2, 4, 3), 255, dtype=np.uint8)
    keep = np.zeros((2, 4), dtype=bool)
    keep[:, :2] = True

    output = hardlock_first_frame(decoded, rgb_first_frame=first, keep_mask=keep)

    assert torch.all(output[:, :, 0, :, :2] == 1)
    assert torch.all(output[:, :, 0, :, 2:] == 0)
    assert torch.all(output[:, :, 1] == 0)
