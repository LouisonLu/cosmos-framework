# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run Cosmos3 PwP Stage1: masked first RGB + masked PCD -> full RGB video.

This is the no-video-path entry point. The model receives a masked first-frame
image, a masked PCD control video, text, and an empty generated RGB placeholder.
The PCD video supplies the output length; a full RGB target video is not needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cosmos_framework.inference.common.init import init_output_dir, init_script

init_script()

import torch

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.inference.pwp_stage1 import (
    hardlock_first_frame,
    load_prompt,
    load_pwp_stage1_inputs,
)
from cosmos_framework.tools.visualize.video import save_img_or_video
from cosmos_framework.utils import log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-file", default=None, help="Required only for an unregistered local DCP checkpoint.")
    parser.add_argument("--experiment", default=None, help="Required only when --config-file is a Python module.")
    parser.add_argument(
        "--first-frame-path",
        required=True,
        help="RGB first-frame image. The spatial mask is applied before it enters the model.",
    )
    parser.add_argument("--pcd-path", required=True, help="PCD control video that defines the 93-frame output.")
    parser.add_argument(
        "--guided-generation-mask-path",
        "--mask-path",
        dest="mask_path",
        default=None,
        help="Binary PNG keep mask: white=keep, black=outpaint. Omit for all-white.",
    )
    parser.add_argument("--prompt-path", required=True, help="Plain text or structured caption JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--no-guardrails",
        action="store_true",
        help="Skip Cosmos Guardrail1 downloads and safety post-processing for this local experiment.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--temporal-interval-mode", choices=("force_one", "max_30fps"), default="force_one")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--mask-fill-value", type=int, default=0)
    hardlock = parser.add_mutually_exclusive_group()
    hardlock.add_argument(
        "--hardlock-first-frame",
        dest="hardlock_first_frame",
        action="store_true",
        help="Restore the first-frame white-mask region exactly after decode (default).",
    )
    hardlock.add_argument(
        "--no-hardlock-first-frame",
        dest="hardlock_first_frame",
        action="store_false",
        help="Keep the model-predicted first frame without decode-space restoration.",
    )
    parser.set_defaults(hardlock_first_frame=True)
    return parser.parse_args()


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().absolute()
    init_output_dir(output_dir)

    setup_kwargs = {
        "checkpoint_path": args.checkpoint_path,
        "output_dir": output_dir,
        "sampler": "unipc",
        "guardrails": not args.no_guardrails,
    }
    if args.config_file is not None:
        setup_kwargs["config_file"] = args.config_file
    if args.experiment is not None:
        setup_kwargs["experiment"] = args.experiment

    log.info("Loading Cosmos3 checkpoint...")
    setup_args = OmniSetupOverrides(**setup_kwargs).build_setup()
    pipe = OmniInference.create(setup_args)

    prompt = load_prompt(args.prompt_path)
    prepared = load_pwp_stage1_inputs(
        first_frame_path=args.first_frame_path,
        pcd_path=args.pcd_path,
        mask_path=args.mask_path,
        prompt=prompt,
        prompt_key=pipe.model.input_caption_key,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        temporal_interval_mode=args.temporal_interval_mode,
        fps=args.fps,
        device=pipe.model.tensor_kwargs["device"],
        mask_fill_value=args.mask_fill_value,
    )
    data_batch = prepared.batch

    (output_dir / "pwp_stage1_input.json").write_text(
        json.dumps(
            {
                "contract": ["masked_first_rgb", "masked_pcd_control", "generated_full_rgb"],
                "mask_role": "stage1_spatial_context_mask",
                "first_frame_path": str(Path(args.first_frame_path).expanduser().absolute()),
                "pcd_path": str(Path(args.pcd_path).expanduser().absolute()),
                "guided_generation_mask_path": (
                    str(Path(args.mask_path).expanduser().absolute()) if args.mask_path else None
                ),
                "prompt_path": str(Path(args.prompt_path).expanduser().absolute()),
                "height": prepared.height,
                "width": prepared.width,
                "num_frames": int(prepared.pcd_frames.shape[0]),
                "fps": prepared.fps,
                "hardlock_first_frame": args.hardlock_first_frame,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info("Generating PwP Stage1 sample...")
    sampler = pipe.model.fixed_step_sampler if pipe.model.config.fixed_step_sampler_config is not None else None
    guidance = 1.0 if sampler is not None else args.guidance
    generated = pipe.model.generate_samples_from_batch(
        data_batch,
        sampler=sampler,
        guidance=guidance,
        seed=[args.seed],
        num_steps=args.num_steps,
        shift=args.shift,
        sigma_max=80.0,
        has_negative_prompt=False,
        n_sample=1,
        normalize_cfg=False,
    )
    latent = generated["vision"][0]
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    decoded = ((1.0 + pipe.model.decode(latent)) / 2.0).clamp(0.0, 1.0)
    generated_frames = int(decoded.shape[2])
    expected_frames = int(prepared.pcd_frames.shape[0])
    if generated_frames != expected_frames:
        raise RuntimeError(f"Cosmos3 decoded {generated_frames} frames, expected {expected_frames}")
    if args.hardlock_first_frame:
        decoded = hardlock_first_frame(
            decoded,
            rgb_first_frame=prepared.first_frame,
            keep_mask=prepared.keep_mask,
        )

    vision_path = output_dir / "vision.mp4"
    save_img_or_video(decoded[0].cpu(), str(vision_path.with_suffix("")), fps=prepared.fps, quality=10)
    (output_dir / "sample_outputs.json").write_text(
        json.dumps({"files": [str(vision_path)], "num_frames": generated_frames}, indent=2),
        encoding="utf-8",
    )
    log.success(f"Saved PwP Stage1 output to '{vision_path}'")
    return vision_path


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
