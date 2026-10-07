import numpy as np

from cosmos_framework.data.generator.local_datasets.pwp_stage1_mask_pool import apply_keep_mask, resize_keep_mask
from cosmos_framework.data.generator.sequence_packing.packers import resolve_item_condition_frames


def test_keep_mask_helper_supports_rgb_and_pcd_shapes() -> None:
    keep = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    pcd = np.stack([rgb, rgb + 1], axis=0)
    masked_rgb = apply_keep_mask(rgb, keep)
    masked_pcd = apply_keep_mask(pcd, keep)
    np.testing.assert_array_equal(masked_rgb[0, 0], rgb[0, 0])
    np.testing.assert_array_equal(masked_rgb[0, 1], [0, 0, 0])
    np.testing.assert_array_equal(masked_pcd[:, 1, 0], [0, 0, 0])


def test_mask_resize_uses_nearest_neighbor() -> None:
    keep = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    resized = resize_keep_mask(keep, 4, 4)
    assert resized.shape == (4, 4)
    assert resized.dtype == np.bool_
    assert resized[0, 0] and not resized[0, 3]


def test_stage1_pack_contract_supervises_only_the_final_rgb_item() -> None:
    assert resolve_item_condition_frames([], item_idx=0, num_items=3, latent_t=1) == [0]
    assert resolve_item_condition_frames([], item_idx=1, num_items=3, latent_t=24) == list(range(24))
    assert resolve_item_condition_frames([], item_idx=2, num_items=3, latent_t=24) == []
