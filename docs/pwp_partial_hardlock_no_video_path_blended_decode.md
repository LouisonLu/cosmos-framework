# Cosmos3 PwP: Partial Hardlock + No Video Path + Blended Decode

An independent Cosmos3 inference pipeline that ports the Cosmos2.5
`inference_partial_hardlock_no_video_path_blended_decode.py` method. It does
not import or change the `pwp_nano_inference*` pipeline, and it does not modify
the core model or sampler.

| Cosmos2.5 file | Cosmos3 file |
|---|---|
| `cosmos_transfer2/config_no_video_path_blended_decode.py` (request schema) | [`cosmos_framework/inference/pwp_partial_hardlock/config.py`](../cosmos_framework/inference/pwp_partial_hardlock/config.py) |
| `inference_pipeline_partial_hardlock_no_video_path.py` (surrogate input, mask erosion) | [`inputs.py`](../cosmos_framework/inference/pwp_partial_hardlock/inputs.py) |
| `vid2vid_model_control_vace_rectified_flow_partial_hardlock.py` (spatial condition mask) | [`latent_hardlock.py`](../cosmos_framework/inference/pwp_partial_hardlock/latent_hardlock.py) |
| `inference_pipeline_partial_hardlock_no_video_path_blended_decode.py` (0°/180° decode) | [`decode.py`](../cosmos_framework/inference/pwp_partial_hardlock/decode.py) |
| `inference_partial_hardlock_no_video_path_blended_decode.py` (orchestration) | [`pipeline.py`](../cosmos_framework/inference/pwp_partial_hardlock/pipeline.py) |
| `examples/inference_partial_hardlock_no_video_path_blended_decode.py` | [`examples/inference_cosmos3_partial_hardlock_no_video_path_blended_decode.py`](../examples/inference_cosmos3_partial_hardlock_no_video_path_blended_decode.py) |

## Contract

```text
model inputs:  masked RGB frame 0 + full PCD video + prompt (+ zero placeholder for the generated item)
not an input:  the RGB video (no video path); in --data-root mode only its frame 0 is extracted
output:        full RGB ERP video, frame 0 hardlocked inside the eroded keep mask
```

These are the same three vision items the PwP Stage 1 and Stage 2 checkpoints
were trained on, so any PwP Nano checkpoint works.

## Method

1. **Masks.** The keep mask's frame 0 (≥128 = keep) masks RGB frame 0 for the
   model. A copy eroded by `guided_generation_mask_erode_px` (default 8, the
   Cosmos2.5 max-pool rule) is resized to the latent grid with
   `align_corners=False` bilinear interpolation, exactly as Cosmos2.5
   `construct_latent_weight_map` does. The masked frame given to the model is
   not eroded. This also matches Cosmos2.5.
2. **Latent partial hardlock.** The reference is the clean latent of the
   masked-frame vision item, which is the latent the model itself conditions on.
   At each UniPC step, inside the locked cells of the generated item's latent
   frame 0, the velocity after CFG is replaced by `(x_t − x0_ref) / σ_t`. Under
   the flow convention `x_t = (1−σ)x0 + σε`, UniPC's x0 prediction there equals
   `x0_ref` exactly, so the state follows the noised reference path and ends at
   `x0_ref`. Soft weights at the mask boundary blend locked and model velocity.
   All other latent frames are generated freely. As in Cosmos2.5 no-video-path
   inference, the lock is active at every step.
3. **Decode.** `blended_decoding=true` (default) decodes at 0° and 180° and
   blends across the ERP seam (two VAE decodes); `false` decodes once.
4. **Optional pixel composite.** `pixel_composite_first_frame=true` also pastes
   the eroded keep region of frame 0 from the source pixels after decode. It is
   off by default, so the output matches Cosmos2.5, whose locked region is the
   VAE reconstruction of the locked latent.

### Difference from the `pwp_nano_inference` temporal hardlock

The `cosmos3_no_video_path_hardlock*` methods in `run_cosmos3_inference_batch.py`
write the clean reference latent into the sampler state after every step. The
network therefore sees clean content labeled with a noisy timestep. This pipeline
keeps the locked cells at the noise level of each step, which is the input
distribution the model was trained on, and still reaches the reference exactly.

## Usage

Single process, one GPU (`CUDA_VISIBLE_DEVICES=<id>`), run from the repo root
after `source .venv/bin/activate && export LD_LIBRARY_PATH=`.

### Data folder

```text
<data-root>/
├── rgb_videos/<stem>.mp4        # only frame 0 is used
├── pcd_videos/<stem>_pcd.mp4
├── prompts/<stem>_prompt.json
└── mask/<stem>_mask.(mp4|png)   # optional; mask_videos/ is also accepted
```

```bash
python examples/inference_cosmos3_partial_hardlock_no_video_path_blended_decode.py \
  --data-root /data/test \
  --checkpoint-path /workspace/hf_model/cosmos3-nano-pwp-stage2-driving2240-1120/iter_000001120_bf16 \
  -o /outputs/stage2_partial_hardlock/test \
  --output-tag cosmos3_stage2_iter1120__partial_hardlock_blended_decode_steps35_guidance6_seed0_erode8 \
  --require-mask --start-index 0 --limit 10 --resume
```

| Option | Default | Meaning |
|---|---|---|
| `--start-index`, `--limit` | 0, all | Range of the sorted samples |
| `--output-tag` | — | Name outputs `<stem>__<tag>.mp4`; otherwise `<stem>_<checkpoint>_<method>.mp4` |
| `--erode-px` | 8 | Hardlock erosion in pixels |
| `--no-blended-decoding` | off | Single VAE decode |
| `--pixel-composite-first-frame` | off | Paste the eroded keep region of frame 0 after decode |
| `--seed`, `--guidance`, `--num-steps`, `--shift` | 0, 6, 35, 5 | Sampling |
| `--resolution`, `--num-frames` | 1024x512, 93 | Output size; frames trimmed to 1 + 4k |
| `--mask-path` | — | One common keep mask for every sample |
| `--resume` | off | Skip samples whose output already exists |
| `--dry-run` | off | Validate inputs without loading the model |

### Request file

`-i requests.json` (one object or a list) or `-i requests.jsonl` (one object
per line). Relative paths resolve against the request file's directory.

```json
{
  "name": "ood__nuscenes__scene-0765__cosmos3_stage1_partial_hardlock",
  "prompt": "A realistic high-quality 360-degree equirectangular video. ...",
  "image_context_path": "frames/ood__nuscenes__scene-0765.png",
  "pcd_path": "pcd_videos/ood__nuscenes__scene-0765_pcd.mp4",
  "guided_generation_mask": "mask/ood__nuscenes__scene-0765_mask.mp4",
  "guided_generation_mask_erode_px": 8,
  "blended_decoding": true,
  "pixel_composite_first_frame": false,
  "seed": 0,
  "guidance": 6.0,
  "num_steps": 35,
  "shift": 5.0
}
```

## Outputs

Per sample, in `-o`:

- `<name>.mp4`: generated video
- `<name>.json`: request and effective settings
- `<name>_input_masked_first_frame.png`: the model's masked frame 0
- `<name>_hardlock_mask.png`: the eroded keep mask used for the latent lock

The run writes `cosmos3_partial_hardlock_no_video_path_blended_decode_summary.json`.
A failed sample is logged and skipped, and the exit code is 1 if any sample failed.
Rerun with `--resume` to retry only the missing outputs.

## Limits

- Requires a non-distilled UniPC checkpoint; it fails clearly on a fixed-step
  (distilled) sampler or a sampler that does not match the UniPC schedule.
- Supports only the PwP three-item vision batch (no other modalities).
- Only frame 0 can be hardlocked: with no video path, later frames have no
  reference.
