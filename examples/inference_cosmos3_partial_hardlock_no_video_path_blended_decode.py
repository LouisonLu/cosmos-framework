# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 PwP inference: latent partial hardlock + no RGB video path + ERP blended decode.

Two input modes; the checkpoint is loaded once and samples run with a progress bar.

Request file (Cosmos2.5-style): ``-i requests.json`` or ``-i requests.jsonl``; each
object follows ``cosmos_framework/inference/pwp_partial_hardlock/config.py``.

Data folder: ``--data-root`` with the Cosmos2.5 / Stage 2 layout::

    rgb_videos/<stem>.mp4          # only frame 0 is used
    pcd_videos/<stem>_pcd.mp4
    prompts/<stem>_prompt.json
    mask/<stem>_mask.(mp4|png)     # optional; mask_videos/ is also accepted

Run as a single process (no torchrun) with one visible GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from tqdm.auto import tqdm

from cosmos_framework.inference.common.init import init_output_dir, init_script

init_script()

from cosmos_framework.inference.pwp_partial_hardlock.config import (  # noqa: E402
    METHOD_NAME,
    PartialHardlockRequest,
    load_requests,
)
from cosmos_framework.inference.pwp_partial_hardlock.inputs import load_prompt  # noqa: E402
from cosmos_framework.utils import log  # noqa: E402


def filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(value.rstrip("/")).name).strip(".-_")
    if not component:
        raise ValueError(f"Cannot derive a filename component from {value!r}")
    return component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", type=Path, help="Request .json or .jsonl")
    source.add_argument("--data-root", type=Path, help="Data folder with rgb_videos/, pcd_videos/, prompts/, mask/")
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument(
        "--checkpoint-label", default=None, help="Default output-name label; defaults to the checkpoint folder"
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--no-ema-weights", dest="use_ema_weights", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Skip samples whose output .mp4 already exists")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list samples without loading the model")

    folder = parser.add_argument_group("--data-root options")
    folder.add_argument("--rgb-dir", default="rgb_videos")
    folder.add_argument("--pcd-dir", default="pcd_videos")
    folder.add_argument("--prompt-dir", default="prompts")
    folder.add_argument("--mask-dir", default="mask")
    folder.add_argument("--pcd-suffix", default="_pcd")
    folder.add_argument("--prompt-suffix", default="_prompt")
    folder.add_argument("--mask-suffix", default="_mask")
    folder.add_argument("--mask-path", type=Path, default=None, help="One common keep mask for every sample")
    folder.add_argument("--require-mask", action="store_true", help="Fail when a sample has no paired mask")
    folder.add_argument("--start-index", type=int, default=0, help="First sample in sorted order")
    folder.add_argument("--limit", type=int, default=None, help="Number of samples from --start-index")
    folder.add_argument("--output-tag", default=None, help="Output name is <stem>__<tag>.mp4")
    folder.add_argument("--ffmpeg-bin", default="ffmpeg")
    folder.add_argument("--erode-px", type=int, default=8)
    folder.add_argument("--no-blended-decoding", dest="blended_decoding", action="store_false")
    folder.add_argument("--pixel-composite-first-frame", action="store_true")
    folder.add_argument("--seed", type=int, default=0)
    folder.add_argument("--guidance", type=float, default=6.0)
    folder.add_argument("--num-steps", type=int, default=35)
    folder.add_argument("--shift", type=float, default=5.0)
    folder.add_argument("--resolution", default="1024x512", help="WIDTHxHEIGHT")
    folder.add_argument("--num-frames", type=int, default=93)
    args = parser.parse_args()
    if args.start_index < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--start-index must be >= 0 and --limit must be positive")
    if not re.fullmatch(r"\d+x\d+", args.resolution):
        parser.error(f"--resolution must be WIDTHxHEIGHT, got {args.resolution!r}")
    return args


def _find_mask(mask_root: Path, stem: str, suffix: str) -> Path | None:
    for name in (f"{stem}{suffix}.png", f"{stem}.png", f"{stem}{suffix}.mp4", f"{stem}.mp4"):
        candidate = mask_root / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def _extract_first_frame(ffmpeg_bin: str, video: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting frame 0 from {video}:\n{completed.stderr[-2000:]}")
    _require_file(destination, f"frame 0 of {video.name}")
    return destination


def requests_from_data_root(args: argparse.Namespace) -> list[tuple[PartialHardlockRequest, Path | None]]:
    """Build one request per RGB video; the second element is the RGB video to extract frame 0 from."""
    root = args.data_root.expanduser().resolve()
    rgb_root, pcd_root, prompt_root = root / args.rgb_dir, root / args.pcd_dir, root / args.prompt_dir
    for path in (rgb_root, pcd_root, prompt_root):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing directory: {path}")
    mask_root = root / args.mask_dir
    if args.mask_dir == "mask" and not mask_root.is_dir() and (root / "mask_videos").is_dir():
        mask_root = root / "mask_videos"
    common_mask = args.mask_path.expanduser().resolve() if args.mask_path is not None else None
    if common_mask is not None:
        _require_file(common_mask, "common mask")

    rgb_videos = sorted(rgb_root.glob("*.mp4"))
    if not rgb_videos:
        raise FileNotFoundError(f"No RGB .mp4 files under {rgb_root}")
    if args.start_index > len(rgb_videos):
        raise ValueError(f"--start-index {args.start_index} is past the {len(rgb_videos)} samples")
    rgb_videos = rgb_videos[args.start_index :]
    if args.limit is not None:
        rgb_videos = rgb_videos[: args.limit]

    width, height = (int(value) for value in args.resolution.split("x"))
    label = filename_component(args.checkpoint_label or args.checkpoint_path)
    work_root = args.output_dir.expanduser().absolute() / "_work"
    requests = []
    for rgb in rgb_videos:
        stem = rgb.stem
        pcd = pcd_root / f"{stem}{args.pcd_suffix}.mp4"
        prompt_path = prompt_root / f"{stem}{args.prompt_suffix}.json"
        _require_file(rgb, f"RGB video for {stem}")
        _require_file(pcd, f"PCD video for {stem}")
        _require_file(prompt_path, f"prompt for {stem}")
        mask = common_mask or (_find_mask(mask_root, stem, args.mask_suffix) if mask_root.is_dir() else None)
        if args.require_mask and mask is None:
            raise FileNotFoundError(f"Missing mask for {stem} under {mask_root}")
        name = (
            f"{filename_component(stem)}__{filename_component(args.output_tag)}"
            if args.output_tag
            else f"{filename_component(stem)}_{label}_{METHOD_NAME}"
        )
        request = PartialHardlockRequest(
            name=name,
            prompt=load_prompt(prompt_path),
            image_context_path=work_root / name / "first_frame.png",
            pcd_path=pcd,
            guided_generation_mask=mask,
            guided_generation_mask_erode_px=args.erode_px,
            blended_decoding=args.blended_decoding,
            pixel_composite_first_frame=args.pixel_composite_first_frame,
            seed=args.seed,
            guidance=args.guidance,
            num_steps=args.num_steps,
            shift=args.shift,
            width=width,
            height=height,
            num_frames=args.num_frames,
        )
        requests.append((request, rgb))
    return requests


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().absolute()
    if args.input is not None:
        jobs: list[tuple[PartialHardlockRequest, Path | None]] = [
            (request, None) for request in load_requests(args.input)
        ]
        for request, _ in jobs:
            _require_file(request.image_context_path, f"image_context_path of {request.name}")
            _require_file(request.pcd_path, f"pcd_path of {request.name}")
            if request.guided_generation_mask is not None:
                _require_file(request.guided_generation_mask, f"guided_generation_mask of {request.name}")
    else:
        jobs = requests_from_data_root(args)

    if args.dry_run:
        print(f"Validated {len(jobs)} sample(s) for {METHOD_NAME}.")
        for request, rgb in jobs:
            print(
                f"{request.name}: frame0={rgb or request.image_context_path} pcd={request.pcd_path} "
                f"mask={request.guided_generation_mask}"
            )
        return 0

    # Imported after --dry-run so validation does not load the model stack.
    from cosmos_framework.inference.pwp_partial_hardlock.pipeline import (
        PartialHardlockNoVideoPathBlendedDecodePipeline,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    init_output_dir(output_dir)
    pipeline = PartialHardlockNoVideoPathBlendedDecodePipeline(
        args.checkpoint_path,
        output_dir,
        config_file=args.config_file,
        experiment=args.experiment,
        use_ema_weights=args.use_ema_weights,
    )

    results = []
    with tqdm(jobs, desc=METHOD_NAME, unit="sample", dynamic_ncols=True) as progress:
        for request, rgb in progress:
            output_path = output_dir / f"{request.name}.mp4"
            progress.set_postfix(sample=request.name[:40], status="running")
            if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
                results.append({"name": request.name, "status": "skipped", "output": str(output_path)})
                progress.set_postfix(sample=request.name[:40], status="skipped")
                continue
            started = time.monotonic()
            log.info(f"Processing sample {request.name}")
            try:
                if rgb is not None:
                    _extract_first_frame(args.ffmpeg_bin, rgb, request.image_context_path)
                pipeline.generate(request, output_dir)
            except Exception as error:  # noqa: BLE001 - record the failure and continue with the next sample.
                log.exception(f"Sample failed: {request.name}")
                results.append({"name": request.name, "status": "failed", "error": str(error)})
                progress.set_postfix(sample=request.name[:40], status="failed")
                continue
            elapsed = time.monotonic() - started
            log.success(f"Saved {output_path} in {elapsed:.1f}s")
            results.append(
                {"name": request.name, "status": "completed", "output": str(output_path), "seconds": elapsed}
            )
            progress.set_postfix(sample=request.name[:40], status="completed")

    summary_path = output_dir / f"{METHOD_NAME}_summary.json"
    summary = {
        "method": METHOD_NAME,
        "checkpoint_path": str(args.checkpoint_path),
        "input": str(args.input or args.data_root),
        "results": results,
        "settings": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    failed = [result["name"] for result in results if result["status"] == "failed"]
    log.info(f"Summary: {summary_path}")
    if failed:
        log.error(f"{len(failed)} sample(s) failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
