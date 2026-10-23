# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dump the pixel-space X/Y sample produced by the Cosmos3 PwP Stage 2 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.local_datasets.helper import save_video_frames_to_mp4
from cosmos_framework.data.generator.local_datasets.pwp_stage1_dataset import _load_jsonl
from cosmos_framework.data.generator.local_datasets.pwp_stage2_dataset import (
    PwPStage2Dataset,
    _prepare_stage2_metadata,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the PwP Stage 2 dataset preprocessing and dump one sample's X/Y."
    )
    parser.add_argument("--manifest", required=True, help="Stage 2 JSONL manifest used by training.")
    parser.add_argument("--output-dir", required=True, help="Directory for tensors, previews, prompt, and contract.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--sample-stem", help="Select the record whose uuid matches this value.")
    selector.add_argument("--index", type=int, help="Zero-based record index in the manifest.")
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true", help="Replace files already present in output-dir.")
    return parser.parse_args()


def _make_visual_only_dataset(args: argparse.Namespace) -> PwPStage2Dataset:
    """Reuse the dataset's actual visual preprocessing without constructing a VLM tokenizer."""
    dataset = PwPStage2Dataset.__new__(PwPStage2Dataset)
    dataset.num_video_frames = args.num_frames
    dataset.target_width = args.width
    dataset.target_height = args.height
    dataset.frame_selection_mode = "first"
    dataset.temporal_interval_mode = "force_one"
    dataset.temporal_compression_factor = 4
    dataset.mask_threshold = args.mask_threshold
    dataset.mask_fill_value = 0
    dataset.cfg_dropout_rate = 0.0
    dataset.caption_suffix = ""
    dataset.conditioning_fps = -1
    # The prompt text is recorded verbatim; tokenization is not needed to inspect X/Y.
    def preserve_prompt(caption: str) -> tuple[list[int], str]:
        return [], caption

    dataset._tokenize_caption = preserve_prompt
    return dataset


def _to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.detach().cpu().float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


def _write_outputs(sample: dict[str, Any], metadata: dict[str, Any], output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        "x_masked_first_rgb.png",
        "x_full_pcd.mp4",
        "x_prompt.txt",
        "y_masked_rgb_gt.mp4",
        "target_loss_mask.mp4",
        "xy_tensors.pt",
        "xy_contract.json",
    )
    existing = [output_dir / name for name in filenames if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs (pass --overwrite): {existing}")

    first_rgb, pcd, target = sample["video"]
    loss_mask = sample["target_loss_mask"].detach().cpu().float()
    fps = float(sample["fps"])

    first_frame = _to_uint8(first_rgb[:, 0]).permute(1, 2, 0).numpy()
    Image.fromarray(first_frame, mode="RGB").save(output_dir / "x_masked_first_rgb.png")
    save_video_frames_to_mp4(_to_uint8(pcd), str(output_dir / "x_full_pcd.mp4"), fps=fps)
    save_video_frames_to_mp4(_to_uint8(target), str(output_dir / "y_masked_rgb_gt.mp4"), fps=fps)

    mask_frames = (loss_mask[0].clamp(0, 1) * 255).round().to(torch.uint8)
    mask_rgb = mask_frames[:, None].expand(-1, 3, -1, -1).permute(1, 0, 2, 3).contiguous()
    save_video_frames_to_mp4(mask_rgb, str(output_dir / "target_loss_mask.mp4"), fps=fps)

    prompt = str(sample["ai_caption"])
    (output_dir / "x_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    tensors = {
        "X": {
            "masked_first_rgb": first_rgb.detach().cpu().contiguous(),
            "pcd_video": pcd.detach().cpu().contiguous(),
            "prompt": prompt,
        },
        "Y": {"masked_rgb_gt": target.detach().cpu().contiguous()},
        "training_auxiliary": {"target_loss_mask": loss_mask.contiguous()},
        "normalization": "video tensors are float32 CTHW in [-1, 1]; loss mask is float32 [1,T,H,W] in {0,1}",
    }
    torch.save(tensors, output_dir / "xy_tensors.pt")

    contract = {
        "sample_uuid": metadata["uuid"],
        "manifest": str(Path(metadata["manifest_path"]).resolve()),
        "source_paths": {
            "rgb": metadata["rgb_path"],
            "pcd": metadata["pcd_path"],
            "mask": metadata["mask_path"],
        },
        "prompt": prompt,
        "preprocessing": {
            "width": int(first_rgb.shape[-1]),
            "height": int(first_rgb.shape[-2]),
            "frames": int(target.shape[1]),
            "frame_start": int(sample["frame_start"]),
            "frame_end": int(sample["frame_end"]),
            "fps": fps,
            "mask_threshold": int(metadata["mask_threshold"]),
            "mask_keep_rule": "mask first channel >= threshold; keep RGB where true; fill other pixels with 0",
            "mask_valid_fraction_per_frame": [round(float(frame.mean().item()), 6) for frame in loss_mask[0]],
        },
        "X": {
            "masked_first_rgb": {
                "tensor_key": "X.masked_first_rgb",
                "shape": list(first_rgb.shape),
                "dtype": str(first_rgb.dtype),
                "normalization": "float32 [-1,1], CTHW; exactly one RGB frame",
                "preview": "x_masked_first_rgb.png",
            },
            "full_pcd_video": {
                "tensor_key": "X.pcd_video",
                "shape": list(pcd.shape),
                "dtype": str(pcd.dtype),
                "normalization": "float32 [-1,1], CTHW",
                "preview": "x_full_pcd.mp4",
            },
            "prompt": {"tensor_key": "X.prompt", "preview": "x_prompt.txt"},
        },
        "Y": {
            "masked_rgb_gt": {
                "tensor_key": "Y.masked_rgb_gt",
                "shape": list(target.shape),
                "dtype": str(target.dtype),
                "normalization": "float32 [-1,1], CTHW",
                "preview": "y_masked_rgb_gt.mp4",
            }
        },
        "training_auxiliary_not_X": {
            "target_loss_mask": {
                "shape": list(loss_mask.shape),
                "valid_fraction_per_frame": [round(float(frame.mean().item()), 6) for frame in loss_mask[0]],
                "preview": "target_loss_mask.mp4",
                "meaning": "1/white contributes to target loss; 0/black is excluded",
            }
        },
        "tokenization_note": (
            "This utility skips VLM tokenizer initialization; prompt text is the exact dataset caption "
            "before Cosmos tokenization."
        ),
    }
    (output_dir / "xy_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    print(f"Dumped Stage 2 sample: {metadata['uuid']}")
    print(f"X masked first RGB: {list(first_rgb.shape)} -> {output_dir / 'x_masked_first_rgb.png'}")
    print(f"X full PCD: {list(pcd.shape)} -> {output_dir / 'x_full_pcd.mp4'}")
    print(f"X prompt: {output_dir / 'x_prompt.txt'}")
    print(f"Y masked RGB GT: {list(target.shape)} -> {output_dir / 'y_masked_rgb_gt.mp4'}")
    print(f"Aux loss mask: {list(loss_mask.shape)} -> {output_dir / 'target_loss_mask.mp4'}")
    print(f"Exact normalized tensors: {output_dir / 'xy_tensors.pt'}")
    print(f"Contract: {output_dir / 'xy_contract.json'}")


def main() -> None:
    args = _parse_args()
    manifest_path = str(Path(args.manifest).resolve())
    records = _prepare_stage2_metadata(_load_jsonl(manifest_path), manifest_path)

    if args.sample_stem is not None:
        matches = [record for record in records if record["uuid"] == args.sample_stem]
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one uuid={args.sample_stem!r}, found {len(matches)}")
        metadata = matches[0]
    else:
        if not 0 <= args.index < len(records):
            raise SystemExit(f"--index must be in [0, {len(records) - 1}], got {args.index}")
        metadata = records[args.index]

    metadata = {**metadata, "manifest_path": manifest_path, "mask_threshold": args.mask_threshold}
    dataset = _make_visual_only_dataset(args)
    sample = dataset.process_one_sample(metadata)
    if sample is None:
        raise SystemExit(f"Stage 2 dataset preprocessing failed for {metadata['uuid']}")
    if len(sample["video"]) != 3:
        raise RuntimeError(
            f"Expected [masked first RGB, full PCD, masked RGB target], got {len(sample['video'])} items"
        )
    if sample["video"][0].shape[1] != 1:
        raise RuntimeError(f"Expected one masked RGB context frame, got shape={tuple(sample['video'][0].shape)}")

    _write_outputs(sample, metadata, Path(args.output_dir).resolve(), args.overwrite)


if __name__ == "__main__":
    main()
