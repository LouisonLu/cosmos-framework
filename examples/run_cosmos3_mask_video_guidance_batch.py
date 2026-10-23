# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Batch Cosmos3 inference from RGB, full PCD, prompt, and a frame-0 mask video.

Directory layout under --data-root::

    rgb_videos/<stem>.mp4
    pcd_videos/<stem>_pcd.mp4
    prompts/<stem>_prompt.json
    mask_videos/<stem>_mask.mp4

For each sample, only mask frame 0 is read. White means keep; black means generate.
The mask is external sampler guidance, not an extra Cosmos3 vision input.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from cosmos_framework.inference.common.init import get_rank, init_output_dir, init_script

init_script()

from cosmos_framework.inference.cosmos3_mask_video_guidance import METHOD_NAME
from cosmos_framework.inference.cosmos3_mask_video_guidance.inputs import discover_samples
from cosmos_framework.inference.cosmos3_mask_video_guidance.pipeline import Cosmos3MaskVideoGuidancePipeline
from cosmos_framework.utils import log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-label", default=None)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--rgb-dir", default="rgb_videos")
    parser.add_argument("--pcd-dir", default="pcd_videos")
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--mask-dir", default="mask_videos")
    parser.add_argument("--pcd-suffix", default="_pcd")
    parser.add_argument("--prompt-suffix", default="_prompt")
    parser.add_argument("--mask-suffix", default="_mask")
    parser.add_argument("--sample-stem", default=None, help="Run exactly one sample with this RGB filename stem")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mask-fill-value", type=int, default=0)
    parser.add_argument(
        "--mask-erode-px",
        type=int,
        default=16,
        help="Erode the white frame-0 keep mask by this many pixels before conditioning and hardlock",
    )
    parser.add_argument("--no-blended-decode", action="store_true")
    parser.add_argument("--no-ema-weights", dest="use_ema_weights", action="store_false")
    parser.set_defaults(use_ema_weights=True)
    return parser.parse_args()


def _label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "checkpoint"


def run(args: argparse.Namespace) -> int:
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("start-index must be non-negative and limit must be positive")
    if min(args.width, args.height, args.num_frames, args.steps) <= 0:
        raise ValueError("width, height, num-frames, and steps must be positive")
    if not 0 <= args.mask_fill_value <= 255:
        raise ValueError("mask-fill-value must be in [0,255]")
    if args.mask_erode_px < 0:
        raise ValueError("mask-erode-px must be non-negative")

    samples = discover_samples(
        args.data_root,
        rgb_dir=args.rgb_dir,
        pcd_dir=args.pcd_dir,
        prompt_dir=args.prompt_dir,
        mask_dir=args.mask_dir,
        pcd_suffix=args.pcd_suffix,
        prompt_suffix=args.prompt_suffix,
        mask_suffix=args.mask_suffix,
    )
    if args.sample_stem is not None:
        samples = [sample for sample in samples if sample.stem == args.sample_stem]
        if len(samples) != 1:
            raise ValueError(f"Expected exactly one sample named {args.sample_stem!r}, found {len(samples)}")
    stop = None if args.limit is None else args.start_index + args.limit
    samples = samples[args.start_index : stop]
    if not samples:
        raise ValueError("No samples selected; check --start-index and --limit")

    output_dir = args.output_dir.expanduser().resolve()
    label = _label(args.checkpoint_label or Path(args.checkpoint_path).name)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "cosmos3_mask_video_guidance_manifest.json"
    if get_rank() == 0:
        manifest_path.write_text(
            json.dumps(
                {
                    "method": METHOD_NAME,
                    "checkpoint": args.checkpoint_path,
                    "mask_erode_px": args.mask_erode_px,
                    "selected_samples": [
                        {
                            "stem": sample.stem,
                            "rgb": str(sample.rgb_path),
                            "pcd": str(sample.pcd_path),
                            "prompt": str(sample.prompt_path),
                            "mask": str(sample.mask_path),
                        }
                        for sample in samples
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.dry_run:
        if get_rank() == 0:
            print(f"Validated {len(samples)} sample(s). Manifest: {manifest_path}")
            for sample in samples:
                print(f"{sample.stem}: RGB={sample.rgb_path} PCD={sample.pcd_path} mask={sample.mask_path}")
        return 0

    init_output_dir(output_dir, resume=args.resume, job_name=METHOD_NAME)
    pipeline = Cosmos3MaskVideoGuidancePipeline(
        args.checkpoint_path,
        output_dir,
        config_file=args.config_file,
        experiment=args.experiment,
        use_ema_weights=args.use_ema_weights,
    )
    if args.num_frames != pipeline.tokenizer.get_pixel_num_frames(
        pipeline.tokenizer.get_latent_num_frames(args.num_frames)
    ):
        raise ValueError("num-frames is incompatible with Cosmos3 VAE temporal compression")

    summary = []
    progress = tqdm(
        samples,
        desc=f"{label}/{METHOD_NAME}",
        unit="sample",
        dynamic_ncols=True,
        disable=get_rank() != 0,
    )
    for index, sample in enumerate(progress):
        seed = args.seed + args.start_index + index
        filename = (
            f"{sample.stem}__cosmos3_{label}_{METHOD_NAME}"
            f"_steps{args.steps}_g{args.guidance:g}_shift{args.shift:g}_seed{seed}"
            f"_erode{args.mask_erode_px}.mp4"
        )
        output_path = output_dir / filename
        metadata_dir = output_dir / "metadata" / Path(filename).stem
        if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
            summary.append({"stem": sample.stem, "status": "skipped", "output": str(output_path)})
            continue

        started = time.monotonic()
        if get_rank() == 0:
            progress.set_postfix(sample=sample.stem[:28], status="running")
            log.info(f"Processing {index + 1}/{len(samples)}: {sample.stem}")
        try:
            pipeline.generate(
                sample,
                output_path,
                metadata_dir,
                width=args.width,
                height=args.height,
                num_frames=args.num_frames,
                fps=args.fps,
                seed=seed,
                num_steps=args.steps,
                guidance=args.guidance,
                shift=args.shift,
                mask_fill_value=args.mask_fill_value,
                mask_erode_px=args.mask_erode_px,
                blended_decode=not args.no_blended_decode,
            )
        except Exception as exc:
            log.exception(f"Sample failed: {sample.stem}")
            summary.append({"stem": sample.stem, "status": "failed", "error": str(exc)})
            if get_rank() == 0:
                progress.set_postfix(sample=sample.stem[:28], status="failed")
            break
        else:
            elapsed = time.monotonic() - started
            summary.append({"stem": sample.stem, "status": "completed", "output": str(output_path)})
            if get_rank() == 0:
                log.success(f"Saved {output_path} in {elapsed:.1f}s")
                progress.set_postfix(sample=sample.stem[:28], status="done")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if get_rank() == 0:
        summary_path = output_dir / "cosmos3_mask_video_guidance_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        failures = [entry for entry in summary if entry["status"] == "failed"]
        completed = len(summary) - len(failures)
        log.info(f"Batch summary: {summary_path} ({completed} completed/skipped, {len(failures)} failed)")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return 1 if any(entry["status"] == "failed" for entry in summary) else 0


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
