# PwP Nano Inference

One Cosmos3 inference pipeline serves every PwP Nano checkpoint. Stage 1 and
Stage 2 describe how checkpoints were trained; they do not select different
inference code paths.

## Input Contract

```text
masked RGB frame 0 + full PCD video + prompt + empty generated-RGB placeholder
                                      -> full RGB video
```

The source RGB video is used only to extract frame 0. The spatial keep mask is
applied to that RGB frame, then reused to hardlock its white region after decode.
PCD is always passed through in full. The method does not pass the RGB video or
the mask as a separate vision item.

## Batch Runner

The sole required command-line runner is
[`examples/run_cosmos3_inference_batch.py`](../examples/run_cosmos3_inference_batch.py).
It discovers aligned RGB, PCD, prompt, and optional mask files under one data
root, preprocesses each sample internally, loads one checkpoint once, shows a
tqdm progress bar over samples, and writes one output video per sample.

```text
<data-root>/
├── rgb_videos/<stem>.mp4
├── pcd_videos/<stem>_pcd.mp4
├── prompts/<stem>_prompt.json
└── mask/<stem>_mask.(png|mp4)   # optional; mask_videos/ is also accepted
```

This is the same layout as the Cosmos2.5 `run_cosmos_inference_batch.py`
runner and the Stage 2 driving dataset. Use `--mask-path` to apply one common
PNG mask to the batch instead of paired masks. When a paired mask is a video,
frame 0 supplies the first-frame mask. Without any mask the full frame 0 is
kept and hardlocked.

| Option | Meaning |
|---|---|
| `--data-root` | Input data folder (layout above; `--layout flat` for one flat folder) |
| `--checkpoint-path` | Cosmos3 checkpoint, e.g. an `iter_*_bf16` export |
| `--method` | Inference method, see below |
| `--output-dir` | Output folder for videos, metadata, manifest, and summary |

```bash
python examples/run_cosmos3_inference_batch.py \
  --data-root /data/driving_dataset/test \
  --checkpoint-path /models/cosmos3-pwp-iter_000001120_bf16 \
  --method cosmos3_no_video_path_hardlock_blended_decode \
  --output-dir /outputs/cosmos3_pwp_stage2_iter1120 \
  --resolution 1024x512 \
  --num-frames 93 \
  --num-steps 35 \
  --shift 5 \
  --seed 0 \
  --resume
```

`--limit 1` runs a single sample. `--dry-run` validates the sample pairs without
loading the model. `--dump-inputs` additionally exports the final prepared
inputs for inspection; a separate preprocessing command is not required.
`python examples/run_cosmos3_inference_batch.py --help` lists every method.

By default, outputs are named
`<stem>_<checkpoint>_<method>.mp4`. Use `--output-tag` to include a custom
checkpoint/method/parameter label. The batch manifest and summary are written
to the output directory.

## Inference Methods

All methods use identical model inputs and sampling; they differ only in the
post-sampling decode and hardlock step.
Implementation: [`cosmos_framework/inference/cosmos3_no_video_path_blended_decode.py`](../cosmos_framework/inference/cosmos3_no_video_path_blended_decode.py).

| `--method` | Decode | Frame-0 hardlock | Cosmos2.5 counterpart |
|---|---|---|---|
| `cosmos3_no_video_path_hardlock_blended_decode` (default) | ERP 0°/180° blended | yes | `inference_partial_hardlock_no_video_path_blended_decode.py` |
| `cosmos3_no_video_path_hardlock` | single | yes | `inference_partial_hardlock_no_video_path.py` |
| `cosmos3_no_video_path` | single | no | — (raw model output, for ablation) |

The Cosmos2.5 script names are accepted as `--method` aliases.

Blended decode:

1. Decode the ERP latent at 0 degrees.
2. Roll it by 180 degrees, decode again, and roll the result back.
3. Blend both decodes with a triangular weight, so the rolled decode covers
   the original left/right ERP seam.

Hardlock replaces the white region of decoded frame 0 with the source RGB
pixels. `--hardlock-erode-px` (default 8, matching the Cosmos2.5 batch
runner) shrinks that region so the mask boundary stays generated.

Difference from Cosmos2.5: Cosmos2.5 hardlocks frame 0 in latent space at every
denoising step through its partial-hardlock condition mask. Cosmos3 condition
masks are per-frame only, so the Cosmos3 hardlock is applied in pixel space
after decode. The blended decode costs two VAE decodes per sample.
