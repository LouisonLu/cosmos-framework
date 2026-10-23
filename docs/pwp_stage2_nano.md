# Cosmos3 PwP Stage 2

This is an independent target-loss-masked continuation of the Cosmos3 PwP
pipeline. It warm-starts the generator from a Stage 1 checkpoint and keeps the
reasoning tower frozen.

## Training contract

Each JSONL record must contain:

```json
{
  "uuid": "scene_0001",
  "rgb_path": "/data/rgb_videos/scene_0001.mp4",
  "pcd_path": "/data/pcd_videos/scene_0001_pcd.mp4",
  "mask_path": "/data/mask/scene_0001_mask.mp4",
  "prompt_path": "/data/prompts/scene_0001_prompt.json"
}
```

`target_loss_mask_path` and `mask_video_path` are accepted as aliases for
`mask_path`.

The dataset emits the following three vision items:

```text
X: masked first RGB, full PCD video, prompt, per-video mask
Y: masked RGB video
```

The mask video is decoded and aligned to the same 93 frames as RGB and PCD.
White pixels (`>= 128`) are valid target RGB. Black pixels are filled with
zero in the RGB target and contribute zero to both the numerator and
denominator of the flow-matching loss. The PCD video is intentionally kept
full and is not masked in this Stage 2 contract.

The pixel mask is conservatively reduced to the RGB target latent grid: a
latent cell is valid only when its source bin contains no black pixel. To
use a one-spatial-latent-stride guard for Cosmos 3, the model also uses a
16-pixel spatial guard, a 1-frame temporal guard, and rejects samples with
less than 1% valid latent coverage. These guards remove boundary-contaminated
latent cells; they do not make black regions contribute to the loss.

## Launch

Set these variables before sourcing the launcher:

```bash
export PWP_STAGE2_MANIFEST=/abs/path/pwp_stage2.jsonl
export PWP_STAGE2_CHECKPOINT_PATH=/abs/path/stage1_checkpoint
export WAN_VAE_PATH=/abs/path/Wan2.2_VAE.pth
export OUTPUT_ROOT=/abs/path/pwp_stage2_output
export NPROC_PER_NODE=8
export MASTER_PORT=50012
source examples/launch_sft_pwp_stage2_nano.sh
```

The manifest mask files are required at dataset construction time. No dynamic
mask-pool sampling or mask generation is performed by this pipeline.
