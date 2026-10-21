#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Cosmos3 PwP Stage 2 diagnostics on one OOD scene with the partial-hardlock
# no-video-path blended-decode pipeline. Each variant changes one thing:
#
#   g1       Stage 2 iter 1120, guidance 1 (no CFG)
#   g3       Stage 2 iter 1120, guidance 3
#   fps30    Stage 2 iter 1120, conditioning fps 30
#   iter280  Stage 2 iter 280  (guidance 7)
#   iter560  Stage 2 iter 560  (guidance 7)
#   iter840  Stage 2 iter 840  (guidance 7)
#   s1g1     Stage 1 iter 2000, guidance 1 (reference)
#
# Usage (from anywhere):
#   bash examples/pwp_stage2_diagnostics.sh            # prepare + run + upload
#   bash examples/pwp_stage2_diagnostics.sh prepare    # environment, checkpoints, VAE, data
#   bash examples/pwp_stage2_diagnostics.sh run        # all variants, spread over GPUs
#   bash examples/pwp_stage2_diagnostics.sh upload     # copy results to Dropbox
#   DIAG_VARIANTS="g1 s1g1" bash examples/pwp_stage2_diagnostics.sh run
#
# Every step is idempotent: existing environments, downloads, and outputs are reused.

set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO/examples/pwp_stage2_diagnostics.sh"
WORK="${WORK:-/workspace/cosmos3_ood_scene0348}"
STEM="${STEM:-ood__nuscenes__scene-0348}"
DATA="$WORK/data/ood"
OUT="$WORK/output/stage2_diagnostics"
VAE_DIR="${VAE_DIR:-/workspace/wan22_vae}"
REMOTE_DATA="dropbox:/World Model (cosmos)/360 video/driving_dataset/ood"
REMOTE_OUT="dropbox:/World Model (cosmos)/360 video/cosmos3/Final Generate Video/cosmos 3 stage2/diagnostics_scene0348"
STAGE1_REPO="LouisonLu/cosmos3-nano-pwp-stage1-maskpool-2000-generator-only-lambdalinear"
STAGE2_REPO="LouisonLu/cosmos3-nano-pwp-stage2-driving2240-1120"
read -r -a VARIANTS <<< "${DIAG_VARIANTS:-g1 g3 fps30 iter280 iter560 iter840 s1g1}"
PY="$REPO/.venv/bin/python"

die() { echo "ERROR: $*" >&2; exit 1; }

# variant -> "stage checkpoint guidance fps"
variant_spec() {
  case "$1" in
    g1) echo "stage2 iter_000001120_bf16 1.0 none" ;;
    g3) echo "stage2 iter_000001120_bf16 3.0 none" ;;
    fps30) echo "stage2 iter_000001120_bf16 7.0 30.0" ;;
    iter280) echo "stage2 iter_000000280_bf16 7.0 none" ;;
    iter560) echo "stage2 iter_000000560_bf16 7.0 none" ;;
    iter840) echo "stage2 iter_000000840_bf16 7.0 none" ;;
    s1g1) echo "stage1 iter_000002000_bf16 1.0 none" ;;
    *) die "unknown variant '$1' (valid: g1 g3 fps30 iter280 iter560 iter840 s1g1)" ;;
  esac
}

ensure_environment() {
  command -v ffmpeg >/dev/null || { apt-get update && apt-get install -y --no-install-recommends ffmpeg; }
  command -v rclone >/dev/null || { apt-get update && apt-get install -y --no-install-recommends rclone; }
  rclone listremotes | grep -qx "dropbox:" || die "rclone has no 'dropbox:' remote; copy ~/.config/rclone/rclone.conf or run 'rclone config'"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found; this needs a GPU machine"
  if [ ! -x "$PY" ]; then
    echo "== Python environment missing; running uv sync (one time)"
    if ! command -v uv >/dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"
    fi
    (cd "$REPO" && uv sync --frozen --extra train --group cu128-train)
  fi
  [ -x "$PY" ] || die "Python environment was not created at $PY"
  test -f "$REPO/examples/inference_cosmos3_partial_hardlock_no_video_path_blended_decode.py" \
    || die "repository is too old; run: git -C $REPO pull --ff-only origin main"
  echo "OK environment: $PY"
}

ensure_checkpoint() {  # stage checkpoint_name
  local stage="$1" name="$2" repo dir
  if [ "$stage" = stage1 ]; then repo="$STAGE1_REPO"; else repo="$STAGE2_REPO"; fi
  dir="$WORK/hf/$stage"
  if [ ! -s "$dir/$name/config.json" ]; then
    echo "== Downloading $repo/$name"
    "$PY" - "$repo" "$name" "$dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, name, local_dir = sys.argv[1:]
snapshot_download(repo_id=repo, allow_patterns=[f"{name}/*"], local_dir=local_dir)
PY
  fi
  [ -s "$dir/$name/config.json" ] || die "checkpoint download failed: $dir/$name"
  # The exported config.json points at the training machine's VAE path; link the local VAE there.
  local vae
  for vae in $(grep -o '"vae_path": "[^"]*"' "$dir/$name/config.json" | cut -d'"' -f4 | sort -u); do
    if [ ! -s "$vae" ]; then
      mkdir -p "$(dirname "$vae")"
      ln -sfn "$VAE_DIR/Wan2.2_VAE.pth" "$vae"
    fi
    [ -s "$vae" ] || die "VAE not available at $vae"
  done
  echo "OK checkpoint: $dir/$name"
}

ensure_vae() {
  if [ ! -s "$VAE_DIR/Wan2.2_VAE.pth" ]; then
    local found
    found="$(find /workspace -type f -name Wan2.2_VAE.pth -size +1M -print -quit 2>/dev/null || true)"
    if [ -n "$found" ]; then
      mkdir -p "$VAE_DIR"
      ln -sfn "$found" "$VAE_DIR/Wan2.2_VAE.pth"
    else
      echo "== Downloading Wan2.2 VAE"
      "$PY" - "$VAE_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    filename="Wan2.2_VAE.pth",
    revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
    local_dir=sys.argv[1],
)
PY
    fi
  fi
  [ -s "$VAE_DIR/Wan2.2_VAE.pth" ] || die "Wan2.2 VAE missing at $VAE_DIR/Wan2.2_VAE.pth"
  echo "OK VAE: $VAE_DIR/Wan2.2_VAE.pth"
}

ensure_data() {
  local item
  for item in "rgb_videos/$STEM.mp4" "pcd_videos/${STEM}_pcd.mp4" "mask/${STEM}_mask.mp4" "prompts/${STEM}_prompt.json"; do
    if [ ! -s "$DATA/$item" ]; then
      rclone copyto "$REMOTE_DATA/$item" "$DATA/$item"
    fi
    [ -s "$DATA/$item" ] || die "missing data file: $DATA/$item"
  done
  echo "OK data: $DATA ($STEM)"
}

prepare() {
  ensure_environment
  ensure_vae
  local v spec
  for v in "${VARIANTS[@]}"; do
    read -r -a spec <<< "$(variant_spec "$v")"
    ensure_checkpoint "${spec[0]}" "${spec[1]}"
  done
  ensure_data
}

run_one() {  # variant gpu
  local v="$1" gpu="$2" spec stage ckpt_name guidance fps ckpt name req frame0
  read -r -a spec <<< "$(variant_spec "$v")"
  stage="${spec[0]}"; ckpt_name="${spec[1]}"; guidance="${spec[2]}"; fps="${spec[3]}"
  ckpt="$WORK/hf/$stage/$ckpt_name"
  [ -s "$ckpt/config.json" ] || die "checkpoint missing: $ckpt (run: bash $SELF prepare)"
  name="${STEM}__cosmos3_${stage}_${ckpt_name%_bf16}_diag_${v}_g${guidance}_fps${fps}_steps35_shift5_seed2025_erode8"
  if [ -s "$OUT/$name.mp4" ]; then echo "SKIP existing $OUT/$name.mp4"; return 0; fi
  mkdir -p "$OUT/_inputs"
  frame0="$OUT/_inputs/${STEM}_first_frame.png"
  [ -s "$frame0" ] || ffmpeg -hide_banner -loglevel error -y -i "$DATA/rgb_videos/$STEM.mp4" -frames:v 1 "$frame0"
  req="$OUT/_inputs/$name.request.json"
  "$PY" - "$req" "$name" "$frame0" "$DATA" "$STEM" "$guidance" "$fps" <<'PY'
import json
import sys

req, name, frame0, data, stem, guidance, fps = sys.argv[1:]
raw = json.load(open(f"{data}/prompts/{stem}_prompt.json", encoding="utf-8"))
prompt = next(raw[k].strip() for k in ("caption", "text", "prompt") if isinstance(raw.get(k), str) and raw[k].strip())
request = {
    "name": name,
    "prompt": prompt,
    "image_context_path": frame0,
    "pcd_path": f"{data}/pcd_videos/{stem}_pcd.mp4",
    "guided_generation_mask": f"{data}/mask/{stem}_mask.mp4",
    "guided_generation_mask_erode_px": 8,
    "blended_decoding": True,
    "seed": 2025,
    "guidance": float(guidance),
    "num_steps": 35,
    "shift": 5.0,
    "width": 1024,
    "height": 512,
    "num_frames": 93,
    "fps": None if fps == "none" else float(fps),
}
json.dump(request, open(req, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
  echo "variant=$v stage=$stage checkpoint=$ckpt guidance=$guidance fps=$fps gpu=$gpu"
  cd "$REPO"
  export PYTHONPATH="$REPO"
  export LD_LIBRARY_PATH=
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/inference_cosmos3_partial_hardlock_no_video_path_blended_decode.py \
    -i "$req" -o "$OUT" --checkpoint-path "$ckpt"
  [ -s "$OUT/$name.mp4" ] || die "no output video for $v"
  echo "PASS $OUT/$name.mp4"
}

run_all() {
  local ngpu g i v
  ngpu="$(nvidia-smi -L | wc -l | tr -d ' ')"
  [ "$ngpu" -ge 1 ] || die "no GPU visible"
  mkdir -p "$OUT/logs"
  echo "== Running ${#VARIANTS[@]} variant(s) on $ngpu GPU(s); logs in $OUT/logs"
  for ((g = 0; g < ngpu && g < ${#VARIANTS[@]}; g++)); do
    (
      for ((i = g; i < ${#VARIANTS[@]}; i += ngpu)); do
        v="${VARIANTS[$i]}"
        if bash "$SELF" _one "$v" "$g" > "$OUT/logs/$v.log" 2>&1; then
          echo "DONE   $v"
        else
          echo "FAILED $v  (full log: $OUT/logs/$v.log)"
          tail -n 15 "$OUT/logs/$v.log" | sed "s/^/   [$v] /"
        fi
      done
    ) &
    [ $((g + 1)) -lt "$ngpu" ] && [ $((g + 1)) -lt ${#VARIANTS[@]} ] && sleep 60
  done
  wait
  echo "== Outputs:"
  ls -1 "$OUT"/*.mp4 2>/dev/null || echo "(none)"
}

upload() {
  local f n count=0
  for f in "$OUT"/*__cosmos3_*_diag_*.mp4 "$OUT"/*__cosmos3_*_diag_*.json; do
    [ -s "$f" ] || continue
    case "$f" in *.request.json) continue ;; esac
    n="${f##*/}"
    rclone copyto "$f" "$REMOTE_OUT/$n" --stats-one-line --stats 10s
    rclone lsf "$REMOTE_OUT" --files-only | grep -Fxq "$n" || die "upload not visible on Dropbox: $n"
    echo "PASS uploaded $n"
    count=$((count + 1))
  done
  echo "== Uploaded $count file(s) to $REMOTE_OUT"
}

for v in "${VARIANTS[@]}"; do variant_spec "$v" >/dev/null; done

case "${1:-all}" in
  prepare) prepare ;;
  run) run_all ;;
  upload) upload ;;
  all) prepare; run_all; upload ;;
  _one) run_one "$2" "$3" ;;
  *) die "usage: bash $SELF [prepare|run|upload|all]" ;;
esac
