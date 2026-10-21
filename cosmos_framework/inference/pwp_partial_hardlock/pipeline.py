# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 PwP inference: latent partial hardlock, no RGB video path, ERP blended decode.

Per sample:

1. Mask RGB frame 0 with the guided-generation keep mask; keep the PCD video full.
2. Sample with UniPC while frame 0 of the generated item is held at the latent of the
   masked frame 0 inside the eroded keep mask (``PartialHardlockSampler``).
3. Decode at 0 and 180 degrees and blend across the ERP seam (or decode once).
4. Optionally paste the eroded keep region of frame 0 from the source pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.inference.pwp_partial_hardlock.config import METHOD_NAME, PartialHardlockRequest
from cosmos_framework.inference.pwp_partial_hardlock.decode import composite_first_frame, decode_erp_blended
from cosmos_framework.inference.pwp_partial_hardlock.inputs import PreparedSample, prepare_sample
from cosmos_framework.inference.pwp_partial_hardlock.latent_hardlock import (
    PartialHardlockSampler,
    VisionItemLayout,
    latent_weight_map,
)
from cosmos_framework.tools.visualize.video import save_img_or_video
from cosmos_framework.utils import log


class PartialHardlockNoVideoPathBlendedDecodePipeline:
    """Load one Cosmos3 checkpoint once and run partial-hardlock requests against it."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        output_dir: Path,
        *,
        config_file: str | None = None,
        experiment: str | None = None,
        use_ema_weights: bool = True,
    ) -> None:
        setup_kwargs: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": output_dir,
            "sampler": "unipc",
            "guardrails": False,  # requests go straight to generate_samples_from_batch
            "use_ema_weights": use_ema_weights,
        }
        if config_file is not None:
            setup_kwargs["config_file"] = config_file
        if experiment is not None:
            setup_kwargs["experiment"] = experiment
        log.info(f"Loading Cosmos3 checkpoint for {METHOD_NAME}: {checkpoint_path}")
        self.pipe = OmniInference.create(OmniSetupOverrides(**setup_kwargs).build_setup())
        self.model = self.pipe.model
        if self.model.config.fixed_step_sampler_config is not None:
            raise ValueError("Partial hardlock requires a non-distilled UniPC checkpoint, not a fixed-step sampler")
        scheduler_type = self.model.config.rectified_flow_inference_config.scheduler_type
        if scheduler_type != "unipc":
            raise ValueError(f"Partial hardlock requires the UniPC scheduler, got {scheduler_type!r}")
        self.tokenizer = self.model.tokenizer_vision_gen

    def _hardlock_sampler(self, prepared: PreparedSample) -> PartialHardlockSampler:
        height, width = prepared.first_frame.shape[:2]
        factor = self.tokenizer.spatial_compression_factor
        if height % factor or width % factor:
            raise ValueError(f"Resolution {width}x{height} must be divisible by the VAE factor {factor}")
        latent_hw = (height // factor, width // factor)
        latent_t = self.tokenizer.get_latent_num_frames(prepared.pcd_frames.shape[0])
        layout = VisionItemLayout(
            channels=self.tokenizer.latent_ch,
            height=latent_hw[0],
            width=latent_hw[1],
            item_frames=(1, latent_t, latent_t),
        )
        return PartialHardlockSampler(
            self.model.sampler,
            layout=layout,
            weight=latent_weight_map(prepared.hardlock_mask, latent_hw),
        )

    @torch.inference_mode()
    def generate(self, request: PartialHardlockRequest, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.name}.mp4"
        prepared = prepare_sample(
            request,
            prompt_key=self.model.input_caption_key,
            temporal_compression=self.tokenizer.temporal_compression_factor,
            device=self.model.tensor_kwargs["device"],
        )
        generated = self.model.generate_samples_from_batch(
            prepared.batch,
            sampler=self._hardlock_sampler(prepared),
            guidance=request.guidance,
            seed=[request.seed],
            num_steps=request.num_steps,
            shift=request.shift,
            sigma_max=80.0,
            has_negative_prompt=False,
            n_sample=1,
            normalize_cfg=False,
        )
        latent = generated["vision"][0]
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        decoded = (
            decode_erp_blended(self.model.decode, latent) if request.blended_decoding else self.model.decode(latent)
        )
        video = ((1.0 + decoded) / 2.0).clamp(0.0, 1.0)
        if request.pixel_composite_first_frame:
            video = composite_first_frame(video, prepared.first_frame, prepared.hardlock_mask)
        if video.shape[2] != prepared.pcd_frames.shape[0]:
            raise RuntimeError(f"Decoded {video.shape[2]} frames, expected {prepared.pcd_frames.shape[0]}")

        save_img_or_video(video[0].cpu(), str(output_path.with_suffix("")), fps=prepared.fps, quality=10)
        self._write_sidecars(request, prepared, output_path)
        del generated, latent, decoded, video
        torch.cuda.empty_cache()
        return output_path

    def _write_sidecars(self, request: PartialHardlockRequest, prepared: PreparedSample, output_path: Path) -> None:
        stem = output_path.with_suffix("")
        Image.fromarray(prepared.masked_first_frame).save(f"{stem}_input_masked_first_frame.png")
        Image.fromarray(prepared.hardlock_mask.astype(np.uint8) * 255).save(f"{stem}_hardlock_mask.png")
        record = {
            "method": METHOD_NAME,
            "request": json.loads(request.model_dump_json()),
            "num_frames": int(prepared.pcd_frames.shape[0]),
            "fps": prepared.fps,
            "contract": ["masked_first_rgb", "full_pcd_control", "generated_rgb", "prompt"],
            "latent_hardlock": "generated item frame 0, eroded keep mask, every UniPC step",
            "decode": "erp_0_180_blended" if request.blended_decoding else "single",
            "pixel_composite_first_frame": request.pixel_composite_first_frame,
        }
        Path(f"{stem}.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
