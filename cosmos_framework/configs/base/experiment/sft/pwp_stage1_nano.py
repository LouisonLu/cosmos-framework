# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""PwP Stage1 Nano generator recipe: masked RGB + masked PCD -> full RGB ERP."""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.configs.base.experiment.sft.vision_sft_nano import vision_sft_nano
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.data.generator.local_datasets.pwp_stage1_dataset import get_pwp_stage1_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L


cs = ConfigStore.instance()


_PWP_STAGE1_NANO_MODEL_CONFIG = copy.deepcopy(NANO_MODEL_CONFIG)
_PWP_STAGE1_NANO_MODEL_CONFIG["action_gen"] = False
_PWP_STAGE1_NANO_MODEL_CONFIG["lora_enabled"] = False
_PWP_STAGE1_NANO_MODEL_CONFIG["resolution"] = "480"
# The two input items are deterministic controls. This keeps them out of the
# scalar flow-matching mean so Stage1 optimizes only the final RGB target.
_PWP_STAGE1_NANO_MODEL_CONFIG["causal_training_strategy"] = "teacher_forcing"


pwp_stage1_nano = copy.deepcopy(vision_sft_nano)
pwp_stage1_nano.job.project = "cosmos3"
pwp_stage1_nano.job.group = "pwp_stage1"
pwp_stage1_nano.job.name = "pwp_stage1_nano_generator_only"
pwp_stage1_nano.model.config = copy.deepcopy(_PWP_STAGE1_NANO_MODEL_CONFIG)
# Cosmos3 is a unified reasoner + generator model. Stage1 updates only the
# generator-side modules; the language/reasoning tower remains frozen.
pwp_stage1_nano.optimizer.keys_to_select = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
]
pwp_stage1_nano.optimizer.lr = 1.0e-4
pwp_stage1_nano.trainer.max_iter = 2000
pwp_stage1_nano.checkpoint.save_iter = 250
pwp_stage1_nano.dataloader_train = L(PackingDataLoader)(
    audio_sample_rate=48000,
    dataset_name="pwp_stage1",
    max_sequence_length=45056,
    patch_spatial=2,
    sound_latent_fps=0,
    tokenizer_spatial_compression_factor=16,
    tokenizer_temporal_compression_factor=4,
    dataloader=L(RankPartitionedDataLoader)(
        batch_size=1,
        in_order=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        sampler=None,
        datasets=dict(
            video=dict(
                ratio=1,
                dataset=L(get_pwp_stage1_dataset)(
                    jsonl_paths=["${oc.env:PWP_STAGE1_MANIFEST}"],
                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                    num_video_frames=93,
                    target_width=1024,
                    target_height=512,
                    frame_selection_mode="first",
                    temporal_interval_mode="force_one",
                    # Keep the fixed prompt present during the single-video
                    # overfit smoke test and deterministic Stage1 protocol.
                    cfg_dropout_rate=0.0,
                    max_caption_tokens=2048,
                    conditioning_fps=-1,
                    temporal_compression_factor=4,
                    mask_pool_root="${oc.env:PWP_STAGE1_MASK_POOL_ROOT}",
                    mask_pool_sampler_py="${oc.env:PWP_STAGE1_MASK_POOL_SAMPLER_PY,}",
                    mask_pool_sampler_seed=0,
                    mask_pool_sampler_kwargs_json="${oc.env:PWP_STAGE1_MASK_POOL_KWARGS,}",
                    mask_fill_value=0,
                ),
            ),
        ),
    ),
)


cs.store(group="experiment", package="_global_", name="pwp_stage1_nano", node=pwp_stage1_nano)


__all__ = ["pwp_stage1_nano"]
