# Cosmos3 Mask-Video Guidance Inference

Run `examples/run_cosmos3_mask_video_guidance_batch.py` with aligned RGB, full
PCD, prompt, and mask-video files. The runner reads mask frame 0, thresholds
white pixels (`>=128`) as keep, then erodes the keep region by 16 pixels by
default (`--mask-erode-px`).

The eroded mask is used consistently to construct the masked RGB first frame,
the frame-0 temporal latent hardlock, and the exact pixel composite after
blended ERP decode. Later temporal mask frames are black, so those frames are
free to generate. The mask is sampler-side guidance, not a fourth learned
vision item. Metadata saves both the source thresholded mask and the effective
eroded mask; output filenames include `_erode16` by default.

```bash
python examples/run_cosmos3_mask_video_guidance_batch.py \
  --data-root /data/ood \
  --mask-dir mask \
  --sample-stem ood__nuscenes__scene-0348 \
  --checkpoint-path /models/stage2/iter_000001120_bf16 \
  --checkpoint-label stage2_iter000001120_lambdalinear \
  --output-dir /outputs/ood_scene0348_stage2 \
  --width 1024 --height 512 --num-frames 93 \
  --steps 35 --guidance 7 --shift 5 --seed 2025 \
  --mask-erode-px 16
```
