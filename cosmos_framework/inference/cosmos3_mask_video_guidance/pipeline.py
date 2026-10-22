# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Standalone Cosmos3 generation pipeline driven by a frame-0 mask video."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.inference.cosmos3_mask_video_guidance import METHOD_NAME
from cosmos_framework.inference.cosmos3_mask_video_guidance.decode import composite_first_frame, decode_erp_blended
from cosmos_framework.inference.cosmos3_mask_video_guidance.inputs import PreparedSample, SamplePaths, prepare_sample
from cosmos_framework.inference.cosmos3_mask_video_guidance.sampler import Frame0MaskGuidanceSampler, PackedVisionLayout
from cosmos_framework.tools.visualize.video import save_img_or_video
from cosmos_framework.utils import log


class Cosmos3MaskVideoGuidancePipeline:
    """Load one Cosmos3 checkpoint and infer batches with frame-0 mask-video guidance."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        output_dir: Path,
        *,
        config_file: str | None = None,
        experiment: str | None = None,
        use_ema_weights: bool = True,
    ) -> None:
        setup: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "output_dir": output_dir,
            "sampler": "unipc",
            "guardrails": False,
            "use_ema_weights": use_ema_weights,
        }
        if config_file is not None:
            setup["config_file"] = config_file
        if experiment is not None:
            setup["experiment"] = experiment
        log.info(f"Loading Cosmos3 checkpoint for {METHOD_NAME}: {checkpoint_path}")
        self.pipe = OmniInference.create(OmniSetupOverrides(**setup).build_setup())
        self.model = self.pipe.model
        if self.model.config.fixed_step_sampler_config is not None:
            raise ValueError("Mask-video latent guidance requires the non-distilled Cosmos3 UniPC sampler")
        scheduler_type = self.model.config.rectified_flow_inference_config.scheduler_type
        if scheduler_type != "unipc":
            raise ValueError(f"Expected the Cosmos3 UniPC scheduler, got {scheduler_type!r}")
        self.tokenizer = self.model.tokenizer_vision_gen

    def _sampler(self, prepared: PreparedSample) -> Frame0MaskGuidanceSampler:
        height, width = prepared.first_frame.shape[:2]
        spatial_factor = self.tokenizer.spatial_compression_factor
        if height % spatial_factor or width % spatial_factor:
            raise ValueError(f"Resolution {width}x{height} must be divisible by VAE factor {spatial_factor}")
        latent_t = self.tokenizer.get_latent_num_frames(prepared.pcd_frames.shape[0])
        layout = PackedVisionLayout(
            channels=self.tokenizer.latent_ch,
            height=height // spatial_factor,
            width=width // spatial_factor,
            item_frames=(1, latent_t, latent_t),
            temporal_compression_factor=self.tokenizer.temporal_compression_factor,
        )
        return Frame0MaskGuidanceSampler(
            self.model.sampler,
            layout=layout,
            temporal_mask=prepared.temporal_guidance_mask,
            reference_item=0,
            target_item=2,
        )

    @torch.inference_mode()
    def generate(
        self,
        sample: SamplePaths,
        output_path: Path,
        metadata_dir: Path,
        *,
        width: int = 1024,
        height: int = 512,
        num_frames: int = 93,
        fps: float | None = None,
        seed: int = 2025,
        num_steps: int = 35,
        guidance: float = 7.0,
        shift: float = 5.0,
        mask_fill_value: int = 0,
        blended_decode: bool = True,
    ) -> Path:
        prepared = prepare_sample(
            sample,
            prompt_key=self.model.input_caption_key,
            num_frames=num_frames,
            temporal_compression=self.tokenizer.temporal_compression_factor,
            width=width,
            height=height,
            mask_fill_value=mask_fill_value,
            fps_override=fps,
            device=self.model.tensor_kwargs["device"],
        )
        generated = self.model.generate_samples_from_batch(
            prepared.batch,
            sampler=self._sampler(prepared),
            guidance=guidance,
            seed=[seed],
            num_steps=num_steps,
            shift=shift,
            sigma_max=80.0,
            has_negative_prompt=False,
            n_sample=1,
            normalize_cfg=False,
        )
        latent = generated["vision"][0]
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        decoded = decode_erp_blended(self.model.decode, latent) if blended_decode else self.model.decode(latent)
        video = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        video = composite_first_frame(video, prepared.first_frame, prepared.spatial_keep_mask)
        if video.shape[2] != num_frames or video.shape[-2:] != (height, width):
            raise RuntimeError(
                f"Decoded shape {tuple(video.shape)}; expected [1,3,{num_frames},{height},{width}]"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        save_img_or_video(video[0].cpu(), str(output_path.with_suffix("")), fps=round(prepared.fps), quality=10)
        self._save_metadata(
            sample,
            prepared,
            output_path,
            metadata_dir,
            blended_decode=blended_decode,
            generation={
                "seed": seed,
                "steps": num_steps,
                "guidance": guidance,
                "shift": shift,
                "mask_fill_value": mask_fill_value,
            },
        )
        del generated, latent, decoded, video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output_path

    @staticmethod
    def _save_metadata(
        sample: SamplePaths,
        prepared: PreparedSample,
        output_path: Path,
        metadata_dir: Path,
        *,
        blended_decode: bool,
        generation: dict[str, Any],
    ) -> None:
        Image.fromarray(prepared.masked_first_frame).save(metadata_dir / "masked_first_rgb.png")
        Image.fromarray(prepared.spatial_keep_mask.astype(np.uint8) * 255).save(metadata_dir / "mask_frame0_keep.png")
        mask_video = np.repeat(prepared.temporal_guidance_mask[..., None].astype(np.uint8) * 255, 3, axis=-1)
        save_img_or_video(
            torch.from_numpy(mask_video).permute(3, 0, 1, 2),
            str(metadata_dir / "temporal_guidance_mask"),
            fps=round(prepared.fps),
            quality=10,
        )
        contract = {
            "method": METHOD_NAME,
            "sample": sample.stem,
            "inputs": {
                "rgb_video": str(sample.rgb_path),
                "pcd_video_full": str(sample.pcd_path),
                "prompt_file": str(sample.prompt_path),
                "mask_video_frame0_source": str(sample.mask_path),
            },
            "model_vision_items": [
                "masked_first_rgb",
                "full_pcd_video",
                "zero_rgb_shape_placeholder_for_generated_target",
            ],
            "prompt": prepared.prompt,
            "target_placeholder": "all-zero pixels define shape only; the sampler initializes the target from noise",
            "mask_video_role": "sampler-side temporal guidance; it is not a fourth learned vision item",
            "temporal_guidance": "mask frame 0 is spatial keep mask; all later frames are black and generated",
            "keep_polarity": "white=keep, black=generate",
            "target_hardlock": "generated RGB target latent frame 0 only, at every UniPC step",
            "output": str(output_path),
            "num_frames": int(prepared.pcd_frames.shape[0]),
            "fps": prepared.fps,
            "resolution": [int(prepared.first_frame.shape[1]), int(prepared.first_frame.shape[0])],
            "decode": "erp_0_180_blended" if blended_decode else "single_vae_decode",
            "first_frame_pixel_composite": "source RGB copied exactly within frame-0 white keep mask",
            "generation": generation,
        }
        (metadata_dir / "model_input_contract.json").write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )
