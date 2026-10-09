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
`model.config.causal_training_strategy=teacher_forcing` so control items are
excluded from the scalar flow-matching mean. Stage1 therefore
computes loss on the complete RGB target, with no target-validity mask.

The mask is sampled once per sample. With the DB235 dynamic sampler, the
default family probabilities are `av2=60%`, `native_fov=10%`, `viewdrop=10%`,
`renderer_confidence=10%`, and `mixed=10%`. Inside `av2`, the
`mask_10_15_percent`, `mask_15_20_percent`, and `mask_20_25_percent` buckets
are equally likely. Files are sampled uniformly within the selected family or
bucket. The same mask is applied to the first RGB frame and every PCD frame;
the RGB target is never masked.

## Inference contract

The independent inference entry point uses the same three-item contract and
does not use the generic `vision_path` video-to-video route:

```text
masked first-frame image + masked PCD video + prompt
                              -> generated full RGB video
```

The spatial guided-generation mask is a binary keep mask: white pixels are
copied into the first-frame and PCD controls, while black pixels are filled
with zero. The generated RGB item is an empty placeholder and is fully
sampled. A full RGB target video is therefore not needed at inference time.

```bash
python examples/inference_pwp_stage1.py \
  --checkpoint-path /models/cosmos3-nano-pwp-stage1-iter100-full \
  --first-frame-path /data/scene_0001_first_frame.png \
  --pcd-path /data/scene_0001_pcd.mp4 \
  --guided-generation-mask-path /data/mask_pool/mask_0001.png \
  --prompt-path /data/prompts/scene_0001_prompt.json \
  --output-dir /outputs/pwp_stage1_iter100_scene_0001 \
  --num-frames 93 \
  --seed 0
```

`--guided-generation-mask-path` may be omitted for an all-white control mask
(`--mask-path` is accepted as an alias). The first-frame white-mask region is
restored after decode by default; use `--no-hardlock-first-frame` to inspect
the raw model prediction. This hardlock is a decode-space blend, separate from
the Stage1 model conditioning and from a Stage2 target-loss mask.

## Batch inference

`examples/inference_pwp_stage1_batch.py` keeps the Cosmos2.5-style directory
layout while loading the Cosmos3 checkpoint only once:

```text
<data-root>/
├── rgb_videos/<stem>.mp4
├── pcd_videos/<stem>_pcd.mp4
└── prompts/<stem>_prompt.json
```

Use `--mask-path` for one common PNG mask. A per-sample `mask_videos/` directory
is also supported with `<stem>_mask.png` or `<stem>_mask.mp4`; a mask video is
converted to its binary frame 0 because the Cosmos3 Stage1 preprocessor takes a
single spatial mask image.

```bash
python examples/inference_pwp_stage1_batch.py \
  --checkpoint-path /models/cosmos3-nano-pwp-stage1-iter2000 \
  --method inference_pwp_stage1 \
  --data-root /data/driving_dataset \
  --output-dir /outputs/pwp_stage1_iter2000 \
  --mask-path /data/mask_pool/mask_common_1024x512.png \
  --num-frames 93 \
  --resolution 1024x512 \
  --num-steps 35 \
  --shift 5 \
  --seed 0 \
  --resume \
  --dump-inputs \
  --no-guardrails
```

Use `--dry-run` to validate all RGB/PCD/prompt pairs without loading the
checkpoint. Each completed sample is written to
`<output-dir>/<stem>_<checkpoint>_<method>.mp4`; `--resume` skips an existing
non-empty video. For example:
`scene_iter_000002000_bf16_inference_pwp_stage1.mp4`. The batch summary and
manifest are written at the output root. Per-sample metadata and the
`--dump-inputs` artifacts are written under
`<output-dir>/metadata/<stem>_<checkpoint>_<method>/`. The progress bar reports
sample completion and the current sample. The current supported method is
`inference_pwp_stage1.py` (also accepted as `pwp_stage1`); `--method` selects
the implementation label and dispatch, rather than executing an arbitrary
Python file.

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

`PWP_STAGE1_MASK_POOL_ROOT` must contain the DB235 train/val family layout with
binary PNG keep masks. The launcher uses the repository's weighted dynamic
sampler by default. Override `PWP_STAGE1_MASK_POOL_SAMPLER_PY` only when using
another compatible sampler. The manifest must not set `mask_path`, because a
record-level mask takes precedence over the pool.

```bash
export PWP_STAGE1_MANIFEST=/data/manifests/pwp_stage1.jsonl
export PWP_STAGE1_MASK_POOL_ROOT=/data/mask_pool
export BASE_CHECKPOINT_PATH=/models/Cosmos3-Nano
export WAN_VAE_PATH=/models/Wan2.2_VAE.pth
export OUTPUT_ROOT=/outputs/cosmos3_pwp_stage1_nano
export NPROC_PER_NODE=8
export PWP_STAGE1_MASK_POOL_KWARGS='{"split":"train"}'

bash examples/launch_sft_pwp_stage1_nano.sh
```

With 2000 records, eight GPUs, per-GPU batch 1, and `grad_accum_iter=1`, the
2000 optimizer iterations represent eight complete passes over the dataset.

The launcher writes the training log below `OUTPUT_ROOT/logs` and checkpoints
according to the isolated Stage1 TOML recipe. Stage2 is intentionally not part
of this pipeline.
