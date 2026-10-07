# PwP Stage1 Nano

This is an isolated Cosmos3 Nano generator-only pipeline for 360 ERP
outpainting. It does not change the existing vision SFT or transfer recipes.

## Training contract

Each sample is packed as three vision items:

1. masked RGB first frame, shape `[3, 1, 512, 1024]`
2. masked PCD/depth control video, shape `[3, 93, 512, 1024]`
3. full RGB target video, shape `[3, 93, 512, 1024]`

The prompt is read from `prompt_path` or `prompt` in the JSONL manifest.
RGB and PCD frames are aligned by frame index. The same spatial mask is
applied to the first RGB frame and every PCD frame. The RGB target remains
complete. The mask uses white as visible input and black as the outpainted
region.

The Cosmos3 sequence plan marks the first two vision items as controls and the
final RGB item as generated. The recipe sets
`model.config.causal_training_strategy=teacher_forcing` so fully-clean control
items are excluded from the scalar flow-matching mean. Stage1 therefore
computes loss on the complete RGB target, with no target-validity mask.

Cosmos3-Nano is the 16B unified reasoner/generator model. The recipe updates
only `moe_gen`, `time_embedder`, `vae2llm`, and `llm2vae`; the reasoning tower
is frozen. This is generator-only training, not LoRA and not full fine-tuning.

For the first single-video overfit check, use the dedicated 100-iteration TOML
and launcher below. It sets `grad_accum_iter=1` to match the historical
Cosmos2.5 run and make the effective eight-GPU count easy to interpret.

## Manifest

Create one JSONL record per aligned RGB/PCD pair:

```json
{"uuid":"scene_0001","rgb_path":"/data/rgb/scene_0001.mp4","pcd_path":"/data/pcd/scene_0001_pcd.mp4","prompt_path":"/data/prompts/scene_0001_prompt.json"}
```

The repository helper creates this format from the legacy folder layout:

```bash
python tools/pwp_stage1_make_manifest.py \
  --rgb-dir /data/rgb_videos \
  --pcd-dir /data/pcd_videos \
  --prompt-dir /data/prompts \
  --output /data/manifests/pwp_stage1.jsonl
```

Validate paths, frame counts, and source geometry before training:

```bash
python tools/pwp_stage1_check_dataset.py /data/manifests/pwp_stage1.jsonl
```

## Launch

`PWP_STAGE1_MASK_POOL_ROOT` must contain binary PNG keep masks. A legacy
dynamic sampler can be supplied with `PWP_STAGE1_MASK_POOL_SAMPLER_PY` and
`PWP_STAGE1_MASK_POOL_KWARGS`.

```bash
export PWP_STAGE1_MANIFEST=/data/manifests/pwp_stage1.jsonl
export PWP_STAGE1_MASK_POOL_ROOT=/data/mask_pool
export BASE_CHECKPOINT_PATH=/models/Cosmos3-Nano
export WAN_VAE_PATH=/models/Wan2.2_VAE.pth
export OUTPUT_ROOT=/outputs/cosmos3_pwp_stage1_nano
export NPROC_PER_NODE=8

bash examples/launch_sft_pwp_stage1_nano.sh
```

The launcher writes the training log below `OUTPUT_ROOT/logs` and checkpoints
according to the isolated Stage1 TOML recipe. Stage2 is intentionally not part
of this pipeline.
