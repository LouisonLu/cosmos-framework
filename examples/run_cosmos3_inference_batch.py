# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Batch Cosmos3 no-video-path inference with full PCD control.

Every sample uses the same contract, regardless of checkpoint or method:

    masked first RGB + full PCD control + prompt + generated zero placeholder

The full RGB video is used only to extract frame 0. It is never passed to the
model as a conditioning video or target during inference. ``--method`` only
changes the post-sampling decode/hardlock step.

Expected layout under ``--data-root`` (``--layout directories``)::

    rgb_videos/<stem>.mp4
    pcd_videos/<stem>_pcd.mp4
    prompts/<stem>_prompt.json
    mask/<stem>_mask.(mp4|png)      # optional; mask_videos/ is also accepted
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from cosmos_framework.inference.common.init import init_output_dir, init_script

init_script()

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.inference.pwp_nano_inference_batch import (
    DEFAULT_INFERENCE_METHOD,
    INFERENCE_METHODS,
    PwPNanoInferenceBatchResult,
    build_output_filename,
    discover_samples,
    filename_component,
    generate_sample,
    resolve_inference_method,
    sample_to_dict,
    select_sample_range,
)
from cosmos_framework.utils import log


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults while keeping the docstring layout and method list readable."""


def parse_args() -> argparse.Namespace:
    methods_help = "\n".join(f"  {name}: {method.description}" for name, method in INFERENCE_METHODS.items())
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"inference methods:\n{methods_help}",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--checkpoint-label",
        default=None,
        help="Filename label; defaults to the checkpoint directory name",
    )
    parser.add_argument(
        "--method",
        "--inference-script",
        dest="method",
        default=DEFAULT_INFERENCE_METHOD,
        help="Inference method (see the list below). Cosmos2.5 script names are accepted as aliases.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional full filename suffix before .mp4; stem is prepended with a double underscore",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--layout", choices=("directories", "flat"), default="directories")
    parser.add_argument("--rgb-dir", default="rgb_videos")
    parser.add_argument("--pcd-dir", default="pcd_videos")
    parser.add_argument("--prompt-dir", default="prompts")
    parser.add_argument("--mask-dir", default="mask")
    parser.add_argument("--pcd-suffix", default="_pcd")
    parser.add_argument("--prompt-suffix", default="_prompt")
    parser.add_argument("--mask-suffix", default="_mask")
    parser.add_argument("--mask-path", type=Path, default=None, help="One common PNG mask for every sample")
    parser.add_argument("--require-mask", action="store_true", help="Fail if any sample has no paired mask")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dump-inputs", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--resolution",
        default=None,
        help="Output resolution as WIDTHxHEIGHT, for example 1024x512",
    )
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--temporal-interval-mode", choices=("force_one", "max_30fps"), default="force_one")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--num-steps", "--steps", dest="num_steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-fill-value", type=int, default=0)
    parser.add_argument(
        "--hardlock-erode-px",
        type=int,
        default=8,
        help="Erode only the frame-0 post-decode hardlock keep region (Cosmos2.5 default: 8); model inputs stay unchanged",
    )
    parser.add_argument("--no-guardrails", action="store_true")
    parser.add_argument("--no-ema-weights", dest="use_ema_weights", action="store_false")
    parser.set_defaults(use_ema_weights=True)
    return parser.parse_args()


def parse_resolution(value: str | None, *, width: int, height: int) -> tuple[int, int]:
    if value is None:
        return width, height
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip().lower())
    if match is None:
        raise ValueError(f"resolution must be WIDTHxHEIGHT, got: {value!r}")
    parsed_width, parsed_height = (int(item) for item in match.groups())
    if parsed_width <= 0 or parsed_height <= 0:
        raise ValueError(f"resolution must be positive, got: {value!r}")
    return parsed_width, parsed_height


@torch.no_grad()
def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = parse_resolution(args.resolution, width=args.width, height=args.height)
    method = resolve_inference_method(args.method)
    method_label = method.name
    checkpoint_label = filename_component(args.checkpoint_label or args.checkpoint_path)
    filename_method = method_label

    if args.hardlock_erode_px < 0:
        raise ValueError("hardlock-erode-px must be non-negative")
    samples = discover_samples(
        args.data_root,
        layout=args.layout,
        rgb_dir=args.rgb_dir,
        pcd_dir=args.pcd_dir,
        prompt_dir=args.prompt_dir,
        mask_dir=args.mask_dir,
        pcd_suffix=args.pcd_suffix,
        prompt_suffix=args.prompt_suffix,
        mask_suffix=args.mask_suffix,
        common_mask=args.mask_path,
        require_mask=args.require_mask,
    )
    samples = select_sample_range(samples, args.start_index, args.limit)

    batch_name = method_label.removeprefix("inference_")
    manifest_path = output_dir / f"{batch_name}_batch_manifest.json"
    manifest_path.write_text(
        json.dumps([sample_to_dict(sample) for sample in samples], indent=2) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(f"Validated {len(samples)} Cosmos3 PwP sample(s) for {method_label} ({method.description}).")
        print(f"Manifest: {manifest_path}")
        for sample in samples:
            print(f"{sample.stem}: rgb={sample.rgb_path} pcd={sample.pcd_path} mask={sample.mask_path}")
        return 0

    init_output_dir(output_dir)
    setup_kwargs = {
        "checkpoint_path": args.checkpoint_path,
        "output_dir": output_dir,
        "sampler": "unipc",
        "guardrails": not args.no_guardrails,
        "use_ema_weights": args.use_ema_weights,
    }
    if args.config_file is not None:
        setup_kwargs["config_file"] = args.config_file
    if args.experiment is not None:
        setup_kwargs["experiment"] = args.experiment

    log.info("Loading Cosmos3 checkpoint once for batch inference...")
    setup_args = OmniSetupOverrides(**setup_kwargs).build_setup()
    pipe = OmniInference.create(setup_args)

    results: list[PwPNanoInferenceBatchResult] = []
    log.info(f"Inference method: {method_label} ({method.description})")
    with tqdm(
        samples,
        desc=f"{checkpoint_label}/{filename_method}",
        unit="sample",
        dynamic_ncols=True,
    ) as progress:
        for index, sample in enumerate(progress):
            output_filename = build_output_filename(
                sample.stem,
                checkpoint_label,
                filename_method,
                output_tag=args.output_tag,
            )
            output_path = output_dir / output_filename
            sample_output_dir = output_dir / "metadata" / Path(output_filename).stem
            sample_work_dir = output_dir / "_work" / Path(output_filename).stem
            progress.set_postfix(sample=sample.stem[:32], status="running")
            if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
                log.info(f"Skipping existing sample {index + 1}/{len(samples)}: {sample.stem}")
                results.append(PwPNanoInferenceBatchResult(sample.stem, "skipped", str(output_path)))
                progress.set_postfix(sample=sample.stem[:32], status="skipped")
                continue

            started = time.monotonic()
            log.info(f"Processing sample {index + 1}/{len(samples)}: {sample.stem}")
            try:
                output_path = generate_sample(
                    pipe,
                    sample,
                    method=method,
                    output_dir=sample_output_dir,
                    output_path=output_path,
                    work_dir=sample_work_dir,
                    height=height,
                    width=width,
                    num_frames=args.num_frames,
                    start_frame=args.start_frame,
                    temporal_interval_mode=args.temporal_interval_mode,
                    fps=args.fps,
                    seed=args.seed,
                    num_steps=args.num_steps,
                    guidance=args.guidance,
                    shift=args.shift,
                    mask_fill_value=args.mask_fill_value,
                    hardlock_erode_px=args.hardlock_erode_px,
                    use_ema_weights=args.use_ema_weights,
                    dump_inputs=args.dump_inputs,
                    ffmpeg_bin=args.ffmpeg_bin,
                )
            except Exception as error:  # noqa: BLE001 - record the sample before aborting the batch.
                elapsed = time.monotonic() - started
                log.exception(f"Sample failed after {elapsed:.1f}s: {sample.stem}")
                results.append(PwPNanoInferenceBatchResult(sample.stem, "failed", error=str(error)))
                progress.set_postfix(sample=sample.stem[:32], status="failed")
                break
            else:
                elapsed = time.monotonic() - started
                log.success(f"Saved {output_path} in {elapsed:.1f}s")
                results.append(PwPNanoInferenceBatchResult(sample.stem, "completed", str(output_path)))
                progress.set_postfix(sample=sample.stem[:32], status="completed")

    summary_path = output_dir / f"{batch_name}_batch_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(Path(args.checkpoint_path).expanduser().absolute()),
                "checkpoint_label": checkpoint_label,
                "method": method_label,
                "data_root": str(args.data_root.expanduser().absolute()),
                "output_dir": str(output_dir),
                "resolution": {"width": width, "height": height},
                "num_samples": len(samples),
                "results": [result.__dict__ for result in results],
                "settings": vars(args),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    failed = [result for result in results if result.status == "failed"]
    log.info(f"Batch summary saved to {summary_path}")
    return 1 if failed else 0


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
