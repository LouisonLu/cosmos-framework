#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Generator-only launcher for the isolated PwP Stage1 pipeline.
# Required: PWP_STAGE1_MANIFEST, PWP_STAGE1_MASK_POOL_ROOT,
#           BASE_CHECKPOINT_PATH, WAN_VAE_PATH.
# Optional: PWP_STAGE1_MASK_POOL_SAMPLER_PY, PWP_STAGE1_MASK_POOL_KWARGS,
#           NPROC_PER_NODE, MASTER_PORT, OUTPUT_ROOT.

TOML_FILE="examples/toml/sft_config/pwp_stage1_nano.toml"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PWP_STAGE1_MANIFEST:?export PWP_STAGE1_MANIFEST=/abs/path/pwp_stage1.jsonl}"
: "${PWP_STAGE1_MASK_POOL_ROOT:?export PWP_STAGE1_MASK_POOL_ROOT=/abs/path/mask_pool}"
: "${BASE_CHECKPOINT_PATH:?export BASE_CHECKPOINT_PATH=/abs/path/Cosmos3-Nano}"
: "${WAN_VAE_PATH:?export WAN_VAE_PATH=/abs/path/Wan2.2_VAE.pth}"
export PWP_STAGE1_MANIFEST PWP_STAGE1_MASK_POOL_ROOT BASE_CHECKPOINT_PATH WAN_VAE_PATH
export PWP_STAGE1_MASK_POOL_SAMPLER_PY="${PWP_STAGE1_MASK_POOL_SAMPLER_PY:-$REPO_ROOT/tools/mask_pool_sampler.py}"
export PWP_STAGE1_MASK_POOL_KWARGS="${PWP_STAGE1_MASK_POOL_KWARGS:-}"

EXTRA_DATASET_CHECK='[[ -f "$PWP_STAGE1_MANIFEST" ]] || {
    echo "ERROR: missing PWP_STAGE1_MANIFEST: $PWP_STAGE1_MANIFEST" >&2
    exit 1
}
[[ -d "$PWP_STAGE1_MASK_POOL_ROOT" ]] || {
    echo "ERROR: missing PWP_STAGE1_MASK_POOL_ROOT: $PWP_STAGE1_MASK_POOL_ROOT" >&2
    exit 1
}'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
