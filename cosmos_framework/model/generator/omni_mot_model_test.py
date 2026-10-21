# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_reasoner_only_setup_skips_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=False,
        lidar_tokenizer=None,
        sound_gen=False,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )
    instantiated = []

    def _instantiate(candidate):
        instantiated.append(candidate)
        return vlm_processor

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert instantiated == [vlm_config.tokenizer]
    assert model.tokenizer_vision_gen is None
    assert model.tokenizer_sound_gen is None


def test_default_setup_loads_vision_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.model.generator import omni_mot_model

    vlm_tokenizer = SimpleNamespace(eos_token_id=42)
    vlm_processor = SimpleNamespace(tokenizer=vlm_tokenizer)
    vision_tokenizer = SimpleNamespace(latent_ch=48, reset_dtype=Mock())
    vlm_config = SimpleNamespace(tokenizer="vlm-tokenizer-config")
    vision_config = SimpleNamespace(temporal_compression_factor=4)
    config = SimpleNamespace(
        load_vision_tokenizer=True,
        lidar_tokenizer=None,
        sound_gen=False,
        state_ch=48,
        tokenizer=vision_config,
        vlm_config=vlm_config,
    )

    def _instantiate(candidate):
        if candidate == vlm_config.tokenizer:
            return vlm_processor
        assert candidate is vision_config
        return vision_tokenizer

    monkeypatch.setattr(omni_mot_model, "lazy_instantiate", _instantiate)
    monkeypatch.setattr(omni_mot_model, "add_special_tokens", lambda tokenizer: (tokenizer, {}))

    model = SimpleNamespace(config=config)
    omni_mot_model.OmniMoTModel.set_up_tokenizers(model)

    assert model.tokenizer_vision_gen is vision_tokenizer
    vision_tokenizer.reset_dtype.assert_called_once_with()


@pytest.mark.parametrize("add_batch_dim", [False, True])
def test_spatial_vision_hardlock_targets_only_frame_zero_of_generated_item(add_batch_dim: bool) -> None:
    import torch

    from cosmos_framework.model.generator.omni_mot_model import _apply_spatial_vision_hardlock_mask

    source = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]])
    pcd = torch.full((1, 2, 2, 2), 99.0)
    target = torch.full((1, 2, 2, 2), -5.0)
    if add_batch_dim:
        source, pcd, target = source.unsqueeze(0), pcd.unsqueeze(0), target.unsqueeze(0)
    clean = SimpleNamespace(
        num_vision_items_per_sample=[3],
        x0_tokens_vision=[source, pcd, target],
        batch_size=1,
    )
    noise = [torch.full((source.numel() + pcd.numel() + target.numel(),), 7.0)]
    reference = [torch.cat((source.flatten(), pcd.flatten(), target.flatten()))]
    condition_mask = [torch.zeros_like(reference[0])]
    temporal_mask = torch.zeros((1, 3, 4, 4), dtype=torch.bool)
    temporal_mask[:, 0, :, :] = True

    updated_noise, updated_reference, updated_mask = _apply_spatial_vision_hardlock_mask(
        temporal_mask,
        clean,
        noise,
        reference,
        condition_mask,
    )

    target_start = source.numel() + pcd.numel()
    target_mask = updated_mask[0][target_start:].reshape_as(target)
    target_noise = updated_noise[0][target_start:].reshape_as(target)
    target_reference = updated_reference[0][target_start:].reshape_as(target)
    source_cthw = source[0] if add_batch_dim else source
    target_mask_cthw = target_mask[0] if add_batch_dim else target_mask
    target_noise_cthw = target_noise[0] if add_batch_dim else target_noise
    target_reference_cthw = target_reference[0] if add_batch_dim else target_reference

    assert torch.equal(target_reference_cthw[:, :1], source_cthw)
    assert torch.all(target_mask_cthw[:, :1] == 1)
    assert torch.all(target_mask_cthw[:, 1:] == 0)
    assert torch.equal(target_noise_cthw[:, :1], source_cthw)
    assert torch.all(target_noise_cthw[:, 1:] == 7)
    assert torch.all(updated_mask[0][:target_start] == 0)


def test_spatial_vision_hardlock_rejects_nonblack_future_masks() -> None:
    import torch

    from cosmos_framework.model.generator.omni_mot_model import _apply_spatial_vision_hardlock_mask

    item = torch.zeros((1, 1, 2, 2))
    clean = SimpleNamespace(num_vision_items_per_sample=[2], x0_tokens_vision=[item, item], batch_size=1)
    state = [torch.zeros(item.numel() * 2)]
    mask = torch.zeros((1, 2, 2, 2), dtype=torch.bool)
    mask[:, 1, 0, 0] = True

    with pytest.raises(ValueError, match="after frame 0 to be black"):
        _apply_spatial_vision_hardlock_mask(mask, clean, state, [state[0].clone()], [state[0].clone()])
