# Historical PwP Stage1 Inference

`examples/inference_pwp_stage1.py` is an isolated restoration of the original
Cosmos3 PwP inference path from commit
`a5877f2278d11cb1bb0eae218cc8a9ce68c41714` (2026-10-08). Its input assembly
does not import or call `pwp_nano_inference`.

## Contract

- Inputs: one RGB first-frame image, a PCD video, a prompt, and an optional
  binary PNG keep mask (white means preserve; black means generate).
- The same spatial mask is applied to the first RGB frame and every PCD frame.
- The model receives masked first RGB, masked PCD, prompt, and an empty RGB
  generation placeholder. It does not receive the RGB target video.
- After decode, frame zero is hardlocked in the white keep region.
- The mask is resized with nearest-neighbor and is not eroded. Decode uses one
  VAE call; this path has no ERP blended decode or temporal latent guidance.

This intentionally retains the historical Stage1 conditioning. Cosmos3 Stage2
training uses full unmasked PCD, so this legacy inference path gives a Stage2
checkpoint a different PCD input distribution than the one used during its
training.
