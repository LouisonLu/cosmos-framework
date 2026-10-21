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
applied to that RGB frame. For hardlock methods, the temporal guidance mask has
the spatial keep mask at frame 0 and is all black for frames 1 onward; it is
projected onto the generated target latent during every sampler step. After
decode, frame 0's white keep region is composited from the source RGB for exact
pixel preservation. PCD is always passed through in full. Neither the RGB video
nor the mask is a separate vision item.

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

All methods use the same model inputs. Hardlock methods apply the temporal
latent mask during sampling, then optionally use blended decoding and exact
pixel-space frame-0 compositing. The raw method disables both hardlock steps.
Latent projection is implemented in
[`omni_mot_model.py`](../cosmos_framework/model/generator/omni_mot_model.py)
and [`unipc.py`](../cosmos_framework/model/generator/diffusion/samplers/unipc.py);
decode and pixel compositing are in
[`cosmos3_no_video_path_blended_decode.py`](../cosmos_framework/inference/cosmos3_no_video_path_blended_decode.py).

| `--method` | Decode | Latent frame-0 hardlock | Pixel frame-0 compositing | Cosmos2.5 counterpart |
|---|---|---|---|---|
| `cosmos3_no_video_path_hardlock_blended_decode` (default) | ERP 0°/180° blended | yes | yes | `inference_partial_hardlock_no_video_path_blended_decode.py` |
| `cosmos3_no_video_path_hardlock` | single | yes | yes | `inference_partial_hardlock_no_video_path.py` |
| `cosmos3_no_video_path` | single | no | no | — (raw model output, for ablation) |

The Cosmos2.5 script names are accepted as `--method` aliases.

Blended decode:

1. Decode the ERP latent at 0 degrees.
2. Roll it by 180 degrees, decode again, and roll the result back.
3. Blend both decodes with a triangular weight, so the rolled decode covers
   the original left/right ERP seam.

The same eroded frame-0 keep mask is used for latent projection and pixel
compositing. `--hardlock-erode-px` (default 8, matching the Cosmos2.5 batch
runner) shrinks that region so the boundary stays generated. The temporal mask
is saved under each sample's `model_inputs/` as an exact `.npz` plus an MP4
preview: frame 0 contains the spatial mask and every later frame is black.

Cosmos2.5 and Cosmos3 now have the same mask behavior: only the white region of
frame 0 is kept, and later frames remain free to generate. Cosmos3 implements
the latent lock by projecting the generated target latent through the resized
mask at every sampler step, then composites frame 0 in pixel space for exact
preservation. The blended decode costs two VAE decodes per sample.
