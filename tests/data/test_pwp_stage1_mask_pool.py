import numpy as np
from PIL import Image

from cosmos_framework.data.generator.local_datasets.pwp_stage1_mask_pool import apply_keep_mask, resize_keep_mask
from cosmos_framework.data.generator.sequence_packing.packers import resolve_item_condition_frames
from tools.mask_pool_sampler import MaskPoolSampler


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


def test_db235_sampler_is_deterministic_and_supports_train_layout(tmp_path) -> None:
    root = tmp_path / "mask_pool"
    families = {
        "native_fov": ["mask.png"],
        "viewdrop": ["mask.png"],
        "renderer_confidence": ["mask.png"],
        "mixed": ["mask.png"],
    }
    for family, filenames in families.items():
        family_dir = root / "train" / family
        family_dir.mkdir(parents=True)
        for filename in filenames:
            Image.fromarray(np.full((512, 1024), 255, dtype=np.uint8)).save(family_dir / filename)
    for bucket in ("mask_10_15_percent", "mask_15_20_percent", "mask_20_25_percent"):
        bucket_dir = root / "train" / "av2" / bucket
        bucket_dir.mkdir(parents=True)
        Image.fromarray(np.full((512, 1024), 255, dtype=np.uint8)).save(bucket_dir / "mask.png")

    first = MaskPoolSampler(root, seed=17)
    second = MaskPoolSampler(root, seed=17)
    first_draws = [first.sample_path()[1] for _ in range(32)]
    second_draws = [second.sample_path()[1] for _ in range(32)]

    assert first_draws == second_draws
    assert {draw["family"] for draw in first_draws}.issubset(
        {"av2", "native_fov", "viewdrop", "renderer_confidence", "mixed"}
    )
    assert all(
        draw["sampling"] in {"weighted_family_then_bucket_uniform_file", "weighted_family_uniform_file"}
        for draw in first_draws
    )


def test_stage1_pack_contract_supervises_only_the_final_rgb_item() -> None:
    assert resolve_item_condition_frames([], item_idx=0, num_items=3, latent_t=1) == [0]
    assert resolve_item_condition_frames([], item_idx=1, num_items=3, latent_t=24) == list(range(24))
    assert resolve_item_condition_frames([], item_idx=2, num_items=3, latent_t=24) == []
