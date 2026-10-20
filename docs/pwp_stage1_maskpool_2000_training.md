# Cosmos 3 PwP Stage 1: Mask-Pool 2000 Run

## Run Record

This document records the completed Cosmos3-Nano Stage 1 run, not just the
repository recipe defaults. The successful run used the later **LambdaLinear**
configuration. Do not confuse it with the earlier Cosmos 3 Stage 1
LambdaCosine run described in [`pwp_lr_training_audit.md`](../../pwp_lr_training_audit.md).

| Item | Successful run |
|---|---|
| Run name | `pwp_stage1_nano_maskpool2000_generator_only_lambdalinear` |
| Initial weights | `/workspace/cosmos3_pwp_stage1_maskpool2000_lambdalinear/base/Cosmos3-Nano` |
| VAE | `/workspace/cosmos3_pwp_stage1_maskpool2000_lambdalinear/base/Wan2.2_VAE.pth` |
| Training data | Dropbox `World Model (cosmos)/360 video/cosmos_train_2000` |
| Dataset | 2,000 aligned RGB / PCD / caption records |
| Mask source | `cosmos_train_2000/mask_pool/train.zip`; 27,957 train PNG masks |
| Frames / resolution | 93 frames, 1024 x 512 |
| Hardware / batch | 8 GPUs, per-GPU batch 1, accumulation 1, global batch 8 |
| Training length | 2,000 optimizer iterations; 8 nominal data passes |
| Result | Training completed successfully; final checkpoint at iteration 2,000 |

The local run directory was:

```text
/workspace/cosmos3_pwp_stage1_maskpool2000_lambdalinear/run_2000/
  cosmos3/pwp_stage1/
  pwp_stage1_nano_maskpool2000_generator_only_lambdalinear/
```

Each nominal pass is `2000 samples / global batch 8 = 250 updates`; therefore
2,000 updates are eight nominal passes. This is a sample-count calculation,
not a claim that each mask repeats or that the dataloader is ordered by epoch.

## Model Contract

```text
X:
  masked RGB first frame
  masked PCD video (same sampled spatial mask on every frame)
  prompt
Y:
  full, unmasked RGB video
Loss:
  generated RGB target only; no target-validity mask
```

The mask is sampled once for each training sample. The same binary 1024x512
mask is used to zero the black region of the first RGB frame and all PCD frames;
the RGB target remains complete. The prompt has `cfg_dropout_rate=0.0`.
The repository dataset implements this contract in
[`pwp_stage1_dataset.py:208`](../cosmos_framework/data/generator/local_datasets/pwp_stage1_dataset.py#L208),
and the mask-pool weights/sampling policy are in
[`mask_pool_sampler.py:4`](../tools/mask_pool_sampler.py#L4) and
[`mask_pool_sampler.py:130`](../tools/mask_pool_sampler.py#L130).

Mask-pool selection was configured with `split=train`, sampler seed `0`, and
weighted family sampling:

| Family | Probability |
|---|---:|
| `av2` | 60% |
| `native_fov` | 10% |
| `viewdrop` | 10% |
| `renderer_confidence` | 10% |
| `mixed` | 10% |

Within `av2`, the `mask_10_15_percent`, `mask_15_20_percent`, and
`mask_20_25_percent` buckets have equal weight. Files are sampled uniformly
inside the selected family/bucket. The extracted archive had `train/` only;
no `val/` directory was required for this run.

Only the generator-side keys `moe_gen`, `time_embedder`, `vae2llm`, and
`llm2vae` were selected for optimization. The reasoning tower was frozen; this
was generator-only training, not full-network fine-tuning or LoRA. The selected
keys are defined in
[`pwp_stage1_nano.py:36`](../cosmos_framework/configs/base/experiment/sft/pwp_stage1_nano.py#L36).

## Effective Training Settings

These are the values checked in the successful run's resolved dry-run and
launch overrides. The source TOML still has different defaults for some fields
(notably scheduler, cycle, and save interval), so use this table for the run
record rather than reading defaults as the executed command.

| Setting | Effective value |
|---|---:|
| Optimizer | AdamW, betas `(0.9, 0.95)`, eps `1e-6`, fused, weight decay `0` |
| Base LR | `1e-4` |
| Scheduler | `LambdaLinear` (`scheduler.lr_scheduler_type=LambdaLinear`) |
| Warmup | 100 updates |
| Cycle length | 5,000 updates |
| `f_start / f_max / f_min` | `1e-6 / 0.5 / 0.2` |
| `trainer.max_iter` | 2,000 |
| `trainer.grad_accum_iter` | 1 |
| Checkpoint interval | 500 updates |
| Training-state load | `false` (fresh optimizer/scheduler state from baseline weights) |
| Strict resume | `false` |

The base LR was not constant at `1e-4`. Approximate effective values for the
LambdaLinear schedule were:

| Point | Effective LR |
|---|---:|
| Update 0 | `1e-10` |
| End of 100-update warmup | `5e-5` |
| Last update (1999) | `3.837e-5` |
| Eventual schedule floor | `2e-5` |

The 5,000-update cycle was longer than this 2,000-update run, so the LR did not
reach its floor. The regular and EMA sample callbacks retained their 5,000-step
defaults and did not run. The successful run ended with exit code 0. The
available resolved-config excerpt showed compile enabled, matching the TOML,
but the final run's saved resolved compile setting was not independently
included in the supplied log; treat compile status as unverified in this
record. The relevant repo defaults are visible in
[`pwp_stage1_nano.toml:38`](../examples/toml/sft_config/pwp_stage1_nano.toml#L38)
and [`pwp_stage1_nano.toml:51`](../examples/toml/sft_config/pwp_stage1_nano.toml#L51).

## Checkpoints And Hugging Face

The DCP training checkpoints were saved at 500, 1,000, 1,500, and 2,000. The
first three were exported as BF16 inference checkpoints. Iteration 2,000 was
exported both as BF16 and as a full DCP continuation checkpoint plus config.
The uploaded repository was verified to contain all four inference exports and
the final full checkpoint.

```text
HF repo: LouisonLu/cosmos3-nano-pwp-stage1-maskpool-2000-generator-only-lambdalinear

iter_000000500_bf16/
iter_000001000_bf16/
iter_000001500_bf16/
iter_000002000_bf16/
iter_000002000_full/
  config.yaml
  checkpoint/
    model/
    optim/
    scheduler/
    trainer/
```

The full 2,000 checkpoint is the one used to initialize Stage 2. Stage 2 loads
its model weights but starts a fresh optimizer/scheduler state; it does not
resume Stage 1's optimizer trajectory. Repository link:
[Hugging Face model](https://huggingface.co/LouisonLu/cosmos3-nano-pwp-stage1-maskpool-2000-generator-only-lambdalinear).

## Reproduction Overrides

The successful Stage 1 run used these key overrides (the environment variables
and eight-GPU launcher supplied the paths and distributed setup):

```bash
TAIL_OVERRIDES=(
  "job.name=pwp_stage1_nano_maskpool2000_generator_only_lambdalinear"
  "optimizer.lr=1e-4"
  "scheduler.lr_scheduler_type=LambdaLinear"
  "scheduler.warm_up_steps=[100]"
  "scheduler.cycle_lengths=[5000]"
  "scheduler.f_start=[1e-6]"
  "scheduler.f_max=[0.5]"
  "scheduler.f_min=[0.2]"
  "trainer.max_iter=2000"
  "trainer.grad_accum_iter=1"
  "checkpoint.save_iter=500"
  "checkpoint.load_training_state=false"
  "checkpoint.strict_resume=false"
)
```

The shared inference method and batch runner are documented in
[`pwp_nano_inference.md`](pwp_nano_inference.md). The training audit and the
earlier LambdaCosine history are in
[`pwp_lr_training_audit.md`](../../pwp_lr_training_audit.md).
