#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Generator-only Cosmos3 PwP Stage 2 launcher.
# Required: PWP_STAGE2_MANIFEST, PWP_STAGE2_CHECKPOINT_PATH, WAN_VAE_PATH.
# The manifest must pair rgb_path, pcd_path, mask_path, and prompt/prompt_path.

TOML_FILE="examples/toml/sft_config/pwp_stage2_nano.toml"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PWP_STAGE2_MANIFEST:?export PWP_STAGE2_MANIFEST=/abs/path/pwp_stage2.jsonl}"
: "${PWP_STAGE2_CHECKPOINT_PATH:?export PWP_STAGE2_CHECKPOINT_PATH=/abs/path/stage1_checkpoint}"
: "${WAN_VAE_PATH:?export WAN_VAE_PATH=/abs/path/Wan2.2_VAE.pth}"
export PWP_STAGE2_MANIFEST PWP_STAGE2_CHECKPOINT_PATH WAN_VAE_PATH
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$PWP_STAGE2_CHECKPOINT_PATH}"

EXTRA_DATASET_CHECK='[[ -f "$PWP_STAGE2_MANIFEST" ]] || {
    echo "ERROR: missing PWP_STAGE2_MANIFEST: $PWP_STAGE2_MANIFEST" >&2
    exit 1
}
[[ -d "$PWP_STAGE2_CHECKPOINT_PATH" ]] || {
    echo "ERROR: missing PWP_STAGE2_CHECKPOINT_PATH: $PWP_STAGE2_CHECKPOINT_PATH" >&2
    exit 1
}'

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
