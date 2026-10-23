import numpy as np
import torch

from cosmos_framework.inference.pwp_stage1_legacy import (
    build_pwp_stage1_batch,
    hardlock_first_frame,
)


def test_legacy_pipeline_masks_first_rgb_and_every_pcd_frame_without_eroding_mask():
    first = np.full((4, 6, 3), 255, dtype=np.uint8)
    pcd = np.full((5, 4, 6, 3), 200, dtype=np.uint8)
    keep = np.zeros((4, 6), dtype=bool)
    keep[:, :3] = True

    batch = build_pwp_stage1_batch(
        first_frame=first,
        pcd_frames=pcd,
        keep_mask=keep,
        prompt="driving scene",
        prompt_key="ai_caption",
        fps=24,
        device="cpu",
    )

    assert batch["num_vision_items_per_sample"] == [3]
    assert torch.all(batch["video"][0][..., :, :3] > 0)
    assert torch.all(batch["video"][0][..., :, 3:] == -1)
    assert torch.all(batch["video"][1][..., :, :3] > 0)
    assert torch.all(batch["video"][1][..., :, 3:] == -1)
    assert torch.all(batch["video"][2] == -1)


def test_legacy_hardlock_restores_only_white_region_of_frame_zero():
    decoded = torch.zeros(1, 3, 2, 2, 4)
    first = np.full((2, 4, 3), 255, dtype=np.uint8)
    keep = np.zeros((2, 4), dtype=bool)
    keep[:, :2] = True

    output = hardlock_first_frame(decoded, rgb_first_frame=first, keep_mask=keep)

    assert torch.all(output[:, :, 0, :, :2] == 1)
    assert torch.all(output[:, :, 0, :, 2:] == 0)
    assert torch.all(output[:, :, 1] == 0)
