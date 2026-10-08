# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dump the exact visual/text inputs prepared by the Cosmos3 PwP Stage1 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.local_datasets.pwp_stage1_mask_pool import apply_keep_mask
from cosmos_framework.inference.pwp_stage1 import load_prompt, load_pwp_stage1_inputs
from cosmos_framework.tools.visualize.video import save_img_or_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-frame-path", required=True)
    parser.add_argument("--pcd-path", required=True)
    parser.add_argument("--guided-generation-mask-path", "--mask-path", dest="mask_path", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--mask-fill-value", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(args.prompt_path)
    prepared = load_pwp_stage1_inputs(
        first_frame_path=args.first_frame_path,
        pcd_path=args.pcd_path,
        mask_path=args.mask_path,
        prompt=prompt,
        prompt_key="ai_caption",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        temporal_interval_mode="force_one",
        fps=args.fps,
        device="cpu",
        mask_fill_value=args.mask_fill_value,
    )

    masked_first = apply_keep_mask(prepared.first_frame, prepared.keep_mask, args.mask_fill_value)
    masked_pcd = apply_keep_mask(prepared.pcd_frames, prepared.keep_mask, args.mask_fill_value)
    generated_placeholder = np.zeros_like(masked_pcd)
    mask_rgb = np.repeat((prepared.keep_mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    mask_video = np.repeat(mask_rgb[None], masked_pcd.shape[0], axis=0)

    Image.fromarray(masked_first, mode="RGB").save(output_dir / "masked_first_frame_model_input.png")
    Image.fromarray(prepared.keep_mask.astype(np.uint8) * 255, mode="L").save(
        output_dir / "guided_generation_mask_resized.png"
    )
    save_img_or_video(
        torch.from_numpy(masked_pcd).permute(3, 0, 1, 2),
        str(output_dir / "masked_pcd_model_input"),
        fps=prepared.fps,
        quality=10,
    )
    save_img_or_video(
        torch.from_numpy(mask_video).permute(3, 0, 1, 2),
        str(output_dir / "guided_generation_mask_video"),
        fps=prepared.fps,
        quality=10,
    )
    save_img_or_video(
        torch.from_numpy(generated_placeholder).permute(3, 0, 1, 2),
        str(output_dir / "generated_rgb_placeholder_model_input"),
        fps=prepared.fps,
        quality=10,
    )

    (output_dir / "prompt_model_input.txt").write_text(prompt + "\n", encoding="utf-8")
    (output_dir / "prompt_model_input.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "source_prompt_path": str(Path(args.prompt_path).expanduser().absolute()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "model_input_contract.json").write_text(
        json.dumps(
            {
                "masked_first_rgb_tensor_shape": [1, 3, 1, prepared.height, prepared.width],
                "masked_pcd_control_tensor_shape": [1, 3, int(prepared.pcd_frames.shape[0]), prepared.height, prepared.width],
                "generated_rgb_placeholder_tensor_shape": [
                    1,
                    3,
                    int(prepared.pcd_frames.shape[0]),
                    prepared.height,
                    prepared.width,
                ],
                "normalization": "uint8 [0,255] -> float32 [-1,1]",
                "mask_semantics": "white=keep control, black=fill with mask_fill_value",
                "mask_fill_value": int(args.mask_fill_value),
                "fps": prepared.fps,
                "num_frames": int(prepared.pcd_frames.shape[0]),
                "prompt_key": "ai_caption",
                "note": "The guide mask is consumed during preprocessing; it is not a separate model vision item.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


if __name__ == "__main__":
    print(run(_parse_args()))
