# Cosmos 3 PwP Stage 2: Driving 2240 / 1120 Run

## Run Record

This document records the completed Stage 2 run using the Stage 1
LambdaLinear checkpoint. It distinguishes the successful launch from earlier
failed attempts and repository defaults.

| Item | Successful run |
|---|---|
| Run name | `pwp_stage2_nano_driving2240_iter1120_generator_only_lambdalinear_no_compile` |
| Training data | Dropbox `World Model (cosmos)/360 video/driving_dataset/train` |
| Dataset | 2,240 aligned RGB / PCD / mask / prompt records; validation passed with zero missing pairs |
| Frames / resolution | 93 frames, 1024 x 512 |
| Hardware / batch | 8 GPUs, per-GPU batch 1, accumulation 1, global batch 8 |
| Training length | 1,120 optimizer iterations; 4 nominal data passes |
| Result | Completed successfully, exit code 0 |
| Elapsed time | About 3 h 44 min 53 sec; progress-reported average 37.96 sec/update |

The local run directory was:

```text
/workspace/cosmos3_pwp_stage2_driving2240_1120/run_stage2_no_compile/
  cosmos3/pwp_stage2/
  pwp_stage2_nano_driving2240_iter1120_generator_only_lambdalinear_no_compile/
```

The four checkpoint save operations took about 106, 102, 119, and 131 seconds
respectively. Dataset arithmetic is `2240 / global batch 8 = 280 updates per
nominal pass`; 1,120 updates are four nominal passes.

## Model And Loss Contract

```text
Model conditioning:
  masked first RGB frame
  full, unmasked PCD video
  prompt

Training-only supervision:
  per-video mask, aligned frame-by-frame to RGB and PCD

Y:
  masked RGB video

Loss:
  white/valid RGB regions contribute
  black holes contribute neither loss numerator nor denominator
```

The mask is an auxiliary `target_loss_mask`, not a fourth visual conditioning
item for the generator. Frame 0 of the mask is used to mask the first RGB
conditioning frame. The same per-frame masks zero the black areas of the RGB
target. PCD stays full and unmasked. The actual dataset emits first masked RGB,
full PCD, and masked target RGB, alongside the auxiliary mask in the batch;
see [`pwp_stage2_dataset.py:145`](../cosmos_framework/data/generator/local_datasets/pwp_stage2_dataset.py#L145).

The dataset thresholds the resized mask at `>=128`, preserves white RGB pixels,
fills black target pixels with zero, and constructs the loss mask on the same
93-frame timeline. The loss implementation multiplies squared error by both
the noisy-target timestep mask and the target-validity mask, then normalizes by
the valid target element count. Thus black pixels are excluded from both
numerator and denominator, rather than treated as black-valued targets. See
[`flow_matching.py:111`](../cosmos_framework/model/generator/algorithm/loss/flow_matching.py#L111)
and [`flow_matching.py:147`](../cosmos_framework/model/generator/algorithm/loss/flow_matching.py#L147).
The pipeline also conservatively maps masks to the latent grid and uses its
spatial/temporal boundary guards; the full contract is in
[`pwp_stage2_nano.md`](pwp_stage2_nano.md).

## Dataset

Remote training root:

```text
dropbox:/World Model (cosmos)/360 video/driving_dataset/train
```

The source had 2,240 RGB files, 2,240 PCD files, 2,240 per-video masks, and
2,240 prompts. Masks were provided in `mask.tar.gz` and prompts in
`prompts.tar.gz`; there was no `mask_videos/` directory at the remote root.
After extraction and manifest construction, all 2,240 records had their RGB,
PCD, mask, and prompt files present.

```text
/workspace/cosmos3_pwp_stage2_driving2240_1120/data/driving_dataset/train/
  rgb_videos/   2240 MP4
  pcd_videos/   2240 MP4
  mask/         2240 extracted mask videos
  prompts/      2240 JSON files

/workspace/cosmos3_pwp_stage2_driving2240_1120/manifests/
  pwp_stage2_driving_train.jsonl   2240 records
```

No dynamic mask-pool sampling was used in Stage 2; each record uses its paired
mask video.

## Initialization And Effective Settings

Stage 2 warm-started from the full Stage 1 checkpoint:

```text
LouisonLu/cosmos3-nano-pwp-stage1-maskpool-2000-generator-only-lambdalinear/
  iter_000002000_full/checkpoint
```

`checkpoint.load_training_state=false` and `checkpoint.strict_resume=false`:
load the Stage 1 model weights, but initialize fresh Stage 2 optimizer,
scheduler, and trainer progress. The same generator-only parameter keys were
updated (`moe_gen`, `time_embedder`, `vae2llm`, `llm2vae`); the reasoning tower
remained frozen.

| Setting | Effective value |
|---|---:|
| Optimizer | AdamW, betas `(0.9, 0.95)`, eps `1e-6`, fused, weight decay `0` |
| Base LR | `2e-5` |
| Scheduler | `LambdaLinear` (`scheduler.lr_scheduler_type=LambdaLinear`) |
| Warmup | 100 updates |
| Cycle length | 1,120 updates |
| `f_start / f_max / f_min` | `1e-6 / 0.5 / 0.2` |
| `trainer.max_iter` | 1,120 |
| `trainer.grad_accum_iter` | 1 |
| Checkpoint interval | 280 updates |
| Compile | Disabled (`model.config.compile.enabled=False`) |
| Online sample callbacks | Existing 5,000-step defaults left unchanged; not reached |

`2e-5` is the base LR, not a constant LR. Approximate effective values:

| Point | Effective LR |
|---|---:|
| Update 0 | `2e-11` |
| End of 100-update warmup | `1e-5` |
| Update 280 | `8.941e-6` |
| Update 560 | `7.294e-6` |
| Update 840 | `5.647e-6` |
| Last update (1119) | `4.006e-6` |
| Schedule endpoint | `4e-6` |

The 1,120-update cycle was intentional and matches the training horizon. This
was the corrected successful run; do not use the earlier failed-run
`cycle_lengths=[1000]` or its LambdaCosine settings. The final launch used
`scheduler.lr_scheduler_type=LambdaLinear`; writing only
`scheduler=lambdalinear` is not an equivalent override in this TOML loader.

## Checkpoints And Hugging Face

The run saved DCP checkpoints at 280, 560, 840, and 1,120. The first three
were uploaded as BF16 inference checkpoints. Iteration 1,120 was uploaded both
as BF16 and as a full continuation checkpoint with its training config. The
last HF verification reported all 109 expected repository files present.

```text
HF repo: LouisonLu/cosmos3-nano-pwp-stage2-driving2240-1120

iter_000000280_bf16/
iter_000000560_bf16/
iter_000000840_bf16/
iter_000001120_bf16/
iter_000001120_full/
  config.yaml
  checkpoint/
    model/
    optim/
    scheduler/
    trainer/
```

The full 1,120 checkpoint is for resuming Stage 2 training; the BF16 folders
are for inference. Repository link:
[Hugging Face model](https://huggingface.co/LouisonLu/cosmos3-nano-pwp-stage2-driving2240-1120).

## Reproduction Overrides

The successful run used these key overrides, including the no-compile setting
that avoided the earlier compile failure:

```bash
TAIL_OVERRIDES=(
  "job.name=pwp_stage2_nano_driving2240_iter1120_generator_only_lambdalinear_no_compile"
  "job.wandb_mode=online"
  "optimizer.lr=2e-5"
  "scheduler.lr_scheduler_type=LambdaLinear"
  "scheduler.warm_up_steps=[100]"
  "scheduler.cycle_lengths=[1120]"
  "scheduler.f_start=[1e-6]"
  "scheduler.f_max=[0.5]"
  "scheduler.f_min=[0.2]"
  "trainer.max_iter=1120"
  "trainer.grad_accum_iter=1"
  "checkpoint.save_iter=280"
  "model.config.compile.enabled=False"
)
```

See [`pwp_stage2_nano.md`](pwp_stage2_nano.md) for the pipeline contract and
launcher requirements, and [`pwp_lr_training_audit.md`](../../pwp_lr_training_audit.md)
for the comparison with Cosmos 2.5 and earlier Cosmos 3 scheduler runs.
