# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""PwP Stage 2 Nano recipe: masked RGB target with validity-masked loss."""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.data.generator.local_datasets.pwp_stage2_dataset import get_pwp_stage2_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L


cs = ConfigStore.instance()


_PWP_STAGE2_NANO_MODEL_CONFIG = copy.deepcopy(NANO_MODEL_CONFIG)
_PWP_STAGE2_NANO_MODEL_CONFIG["action_gen"] = False
_PWP_STAGE2_NANO_MODEL_CONFIG["lora_enabled"] = False
_PWP_STAGE2_NANO_MODEL_CONFIG["resolution"] = "480"
_PWP_STAGE2_NANO_MODEL_CONFIG["causal_training_strategy"] = "teacher_forcing"


pwp_stage2_nano = copy.deepcopy(vision_sft_nano)
pwp_stage2_nano.job.project = "cosmos3"
pwp_stage2_nano.job.group = "pwp_stage2"
pwp_stage2_nano.job.name = "pwp_stage2_nano_target_loss_mask"
pwp_stage2_nano.model.config = copy.deepcopy(_PWP_STAGE2_NANO_MODEL_CONFIG)

# Stage 2 warm-starts the generator from Stage 1. The reasoning tower remains
# frozen, matching the validated generator-only Stage 1 recipe.
pwp_stage2_nano.optimizer.keys_to_select = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
]
pwp_stage2_nano.optimizer.lr = 1.0e-4
pwp_stage2_nano.trainer.grad_accum_iter = 1
pwp_stage2_nano.trainer.max_iter = 2000
pwp_stage2_nano.checkpoint.load_path = "${oc.env:PWP_STAGE2_CHECKPOINT_PATH}"
pwp_stage2_nano.checkpoint.load_training_state = False
pwp_stage2_nano.checkpoint.strict_resume = False
pwp_stage2_nano.checkpoint.save_iter = 250

pwp_stage2_nano.dataloader_train = L(PackingDataLoader)(
    audio_sample_rate=48000,
    dataset_name="pwp_stage2",
    max_sequence_length=45056,
    patch_spatial=2,
    sound_latent_fps=0,
    tokenizer_spatial_compression_factor=16,
    tokenizer_temporal_compression_factor=4,
    dataloader=L(RankPartitionedDataLoader)(
        batch_size=1,
        in_order=True,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=None,
        sampler=None,
        datasets=dict(
            video=dict(
                ratio=1,
                dataset=L(get_pwp_stage2_dataset)(
                    jsonl_paths=["${oc.env:PWP_STAGE2_MANIFEST}"],
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                    num_video_frames=93,
                    target_width=1024,
                    target_height=512,
                    frame_selection_mode="first",
                    temporal_interval_mode="force_one",
                    cfg_dropout_rate=0.0,
                    max_caption_tokens=2048,
                    conditioning_fps=-1,
                    temporal_compression_factor=4,
                    mask_threshold=128,
                    mask_fill_value=0,
                ),
            ),
        ),
    ),
)


cs.store(group="experiment", package="_global_", name="pwp_stage2_nano", node=pwp_stage2_nano)


__all__ = ["pwp_stage2_nano"]
