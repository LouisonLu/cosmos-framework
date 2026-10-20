# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Batch helpers for Cosmos3 no-video-path inference with full PCD control."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.local_datasets.pwp_stage1_mask_pool import apply_keep_mask
from cosmos_framework.inference.pwp_nano_inference import (
    PwPNanoPreparedInputs,
    load_prompt,
    load_pwp_nano_inference_inputs,
)
from cosmos_framework.inference.cosmos3_no_video_path_blended_decode import decode_erp_blended, hardlock_first_frame
from cosmos_framework.tools.visualize.video import save_img_or_video


@dataclass(frozen=True)
class PwPNanoInferenceSample:
    """One paired RGB/PCD/prompt inference sample."""

    stem: str
    rgb_path: Path
    pcd_path: Path
    prompt_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class PwPNanoInferenceBatchResult:
    stem: str
    status: str
    output_path: str | None = None
    error: str | None = None


def filename_component(value: str) -> str:
    """Convert a path or label into one stable filename component."""

    value = value.rstrip("/")
    component = Path(value).stem if value.endswith(".py") else Path(value).name
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", component).strip(".-_")
    if not component:
        raise ValueError(f"Cannot derive a filename component from: {value!r}")
    return component


@dataclass(frozen=True)
class InferenceMethod:
    """One no-video-path inference variant; all share the same model inputs."""

    name: str
    blended_decode: bool
    hardlock_first_frame: bool
    description: str


DEFAULT_INFERENCE_METHOD = "cosmos3_no_video_path_hardlock_blended_decode"

INFERENCE_METHODS: dict[str, InferenceMethod] = {
    method.name: method
    for method in (
        InferenceMethod(
            "cosmos3_no_video_path_hardlock_blended_decode",
            blended_decode=True,
            hardlock_first_frame=True,
            description="ERP 0/180-degree blended decode, then frame-0 white-region hardlock",
        ),
        InferenceMethod(
            "cosmos3_no_video_path_hardlock",
            blended_decode=False,
            hardlock_first_frame=True,
            description="single VAE decode, then frame-0 white-region hardlock",
        ),
        InferenceMethod(
            "cosmos3_no_video_path",
            blended_decode=False,
            hardlock_first_frame=False,
            description="single VAE decode, no hardlock (raw model output)",
        ),
    )
}

# Cosmos2.5 entry-point names map to their Cosmos3 counterparts.
_METHOD_ALIASES = {
    "cosmos3_no_video_path_blended_decode": "cosmos3_no_video_path_hardlock_blended_decode",
    "inference_partial_hardlock_no_video_path_blended_decode": "cosmos3_no_video_path_hardlock_blended_decode",
    "inference_partial_hardlock_no_video_path": "cosmos3_no_video_path_hardlock",
}


def resolve_inference_method(value: str) -> InferenceMethod:
    """Resolve a method name, Cosmos2.5 script name, or script path to a method."""
    method = filename_component(value)
    method = _METHOD_ALIASES.get(method, method)
    try:
        return INFERENCE_METHODS[method]
    except KeyError as error:
        expected = ", ".join(INFERENCE_METHODS)
        raise ValueError(f"Unsupported inference method: {value!r}; expected one of: {expected}.") from error


def build_output_filename(
    stem: str,
    checkpoint_label: str,
    method: str,
    output_tag: str | None = None,
) -> str:
    """Build a stable filename, optionally using an explicit metadata-rich tag."""

    if output_tag:
        return f"{filename_component(stem)}__{filename_component(output_tag)}.mp4"

    return (
        f"{filename_component(stem)}_"
        f"{filename_component(checkpoint_label)}_"
        f"{filename_component(method)}.mp4"
    )


def select_sample_range(
    samples: list[PwPNanoInferenceSample],
    start_index: int,
    limit: int | None,
) -> list[PwPNanoInferenceSample]:
    """Select a bounded range from the full, deterministically sorted manifest."""
    if start_index < 0 or start_index > len(samples):
        raise ValueError(f"start_index {start_index} is outside {len(samples)} samples")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected = samples[start_index:]
    return selected if limit is None else selected[:limit]


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")


def _parse_prompt(path: Path) -> str:
    prompt = load_prompt(path)
    if not prompt.strip():
        raise ValueError(f"Prompt is empty: {path}")
    return prompt


def _find_mask(mask_root: Path, stem: str, mask_suffix: str) -> Path | None:
    """Resolve a static PNG or a video mask using the Cosmos2.5 naming convention."""

    candidates = (
        mask_root / f"{stem}{mask_suffix}.png",
        mask_root / f"{stem}.png",
        mask_root / f"{stem}{mask_suffix}.mp4",
        mask_root / f"{stem}.mp4",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def discover_samples(
    data_root: Path,
    *,
    layout: str = "directories",
    rgb_dir: str = "rgb_videos",
    pcd_dir: str = "pcd_videos",
    prompt_dir: str = "prompts",
    mask_dir: str = "mask",
    pcd_suffix: str = "_pcd",
    prompt_suffix: str = "_prompt",
    mask_suffix: str = "_mask",
    common_mask: Path | None = None,
    limit: int | None = None,
    require_mask: bool = False,
) -> list[PwPNanoInferenceSample]:
    """Discover and validate samples without loading the model."""

    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    if layout == "directories":
        rgb_root = data_root / rgb_dir
        pcd_root = data_root / pcd_dir
        prompt_root = data_root / prompt_dir
        mask_root = data_root / mask_dir
        # Cosmos2.5 and the Stage 2 data use mask/; older Cosmos3 batches used mask_videos/.
        if mask_dir == "mask" and not mask_root.is_dir() and (data_root / "mask_videos").is_dir():
            mask_root = data_root / "mask_videos"
        for path, label in (
            (rgb_root, "RGB directory"),
            (pcd_root, "PCD directory"),
            (prompt_root, "prompt directory"),
        ):
            if not path.is_dir():
                raise FileNotFoundError(f"Missing {label}: {path}")
        rgb_paths = sorted(rgb_root.glob("*.mp4"))
        pcd_for = lambda stem: pcd_root / f"{stem}{pcd_suffix}.mp4"
        prompt_for = lambda stem: prompt_root / f"{stem}{prompt_suffix}.json"
    elif layout == "flat":
        rgb_paths = sorted(
            path
            for path in data_root.glob("*.mp4")
            if not path.name.endswith(f"{pcd_suffix}.mp4")
            and not path.name.endswith(f"{mask_suffix}.mp4")
        )
        mask_root = data_root
        pcd_for = lambda stem: data_root / f"{stem}{pcd_suffix}.mp4"
        prompt_for = lambda stem: data_root / f"{stem}{prompt_suffix}.json"
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if not rgb_paths:
        raise FileNotFoundError(f"No RGB .mp4 files found under: {data_root}")
    if common_mask is not None:
        common_mask = common_mask.expanduser().resolve()
        _require_file(common_mask, "common mask")

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rgb_paths = rgb_paths[:limit]

    samples: list[PwPNanoInferenceSample] = []
    for rgb_path in rgb_paths:
        stem = rgb_path.stem
        pcd_path = pcd_for(stem)
        prompt_path = prompt_for(stem)
        _require_file(rgb_path, f"RGB video for {stem}")
        _require_file(pcd_path, f"PCD video for {stem}")
        _require_file(prompt_path, f"prompt for {stem}")
        _parse_prompt(prompt_path)

        mask_path = common_mask
        if mask_path is None and layout == "directories":
            mask_path = _find_mask(mask_root, stem, mask_suffix)
        if require_mask and mask_path is None:
            raise FileNotFoundError(f"Missing per-sample mask for {stem} under {mask_root}")

        samples.append(
            PwPNanoInferenceSample(
                stem=stem,
                rgb_path=rgb_path,
                pcd_path=pcd_path,
                prompt_path=prompt_path,
                mask_path=mask_path,
            )
        )
    return samples


def extract_first_frame(video_path: Path, output_path: Path, ffmpeg_bin: str = "ffmpeg") -> Path:
    """Extract RGB frame 0 for the image-context input."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting frame 0 from {video_path}:\n{completed.stderr[-2000:]}"
        )
    _require_file(output_path, f"first frame for {video_path.stem}")
    return output_path


def resolve_mask_for_inference(
    mask_path: Path | None,
    *,
    work_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
) -> Path | None:
    """Convert a per-sample mask video to the PNG accepted by Cosmos3."""

    if mask_path is None or mask_path.suffix.lower() != ".mp4":
        return mask_path
    mask_png = work_dir / "mask_frame0.png"
    extract_first_frame(mask_path, mask_png, ffmpeg_bin=ffmpeg_bin)
    with Image.open(mask_png) as image:
        binary = image.convert("L").point(lambda value: 255 if value >= 128 else 0)
        binary.save(mask_png)
    return mask_png


def dump_prepared_inputs(
    prepared: PwPNanoPreparedInputs,
    *,
    output_dir: Path,
    prompt: str,
    first_frame_path: Path,
    pcd_path: Path,
    prompt_path: Path,
    mask_path: Path | None,
) -> None:
    """Write human-inspectable versions of the exact pre-model tensors."""

    output_dir.mkdir(parents=True, exist_ok=True)
    masked_first = apply_keep_mask(prepared.first_frame, prepared.keep_mask, 0)
    pcd_control = prepared.pcd_frames
    generated_placeholder = np.zeros_like(pcd_control)
    mask_rgb = np.repeat((prepared.keep_mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
    mask_video = np.repeat(mask_rgb[None], pcd_control.shape[0], axis=0)

    Image.fromarray(masked_first, mode="RGB").save(output_dir / "masked_first_frame_model_input.png")
    Image.fromarray(prepared.keep_mask.astype(np.uint8) * 255, mode="L").save(
        output_dir / "guided_generation_mask_resized.png"
    )
    save_img_or_video(
        torch.from_numpy(pcd_control).permute(3, 0, 1, 2),
        str(output_dir / "full_pcd_model_input"),
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
                "source_prompt_path": str(prompt_path.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "model_input_contract.json").write_text(
        json.dumps(
            {
                "contract": [
                    "masked_first_rgb",
                    "full_pcd_control",
                    "generated_rgb_placeholder",
                    "prompt",
                ],
                "first_frame_source": str(first_frame_path.resolve()),
                "pcd_source": str(pcd_path.resolve()),
                "guided_generation_mask_source": str(mask_path.resolve()) if mask_path else None,
                "masked_first_rgb_tensor_shape": [1, 3, 1, prepared.height, prepared.width],
                "pcd_control_tensor_shape": [
                    1,
                    3,
                    int(prepared.pcd_frames.shape[0]),
                    prepared.height,
                    prepared.width,
                ],
                "generated_rgb_placeholder_tensor_shape": [
                    1,
                    3,
                    int(prepared.pcd_frames.shape[0]),
                    prepared.height,
                    prepared.width,
                ],
                "normalization": "uint8 [0,255] -> float32 [-1,1]",
                "mask_semantics": (
                    "mask first RGB and hardlock the white frame-0 region; PCD remains full"
                ),
                "mask_fill_value": 0,
                "fps": prepared.fps,
                "num_frames": int(prepared.pcd_frames.shape[0]),
                "note": "guided_generation_mask is consumed during preprocessing; it is not a fourth vision item.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@torch.inference_mode()
def generate_sample(
    pipe: Any,
    sample: PwPNanoInferenceSample,
    *,
    method: InferenceMethod,
    output_dir: Path,
    output_path: Path,
    work_dir: Path,
    height: int,
    width: int,
    num_frames: int,
    start_frame: int,
    temporal_interval_mode: str,
    fps: float | None,
    seed: int,
    num_steps: int,
    guidance: float,
    shift: float,
    mask_fill_value: int,
    hardlock_erode_px: int,
    use_ema_weights: bool,
    dump_inputs: bool,
    ffmpeg_bin: str,
) -> Path:
    """Run one sample using an already-loaded Cosmos3 pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    first_frame_path = extract_first_frame(sample.rgb_path, work_dir / "first_frame.png", ffmpeg_bin)
    mask_path = resolve_mask_for_inference(sample.mask_path, work_dir=work_dir, ffmpeg_bin=ffmpeg_bin)
    prompt = load_prompt(sample.prompt_path)

    prepared = load_pwp_nano_inference_inputs(
        first_frame_path=first_frame_path,
        pcd_path=sample.pcd_path,
        mask_path=mask_path,
        prompt=prompt,
        prompt_key=pipe.model.input_caption_key,
        height=height,
        width=width,
        num_frames=num_frames,
        start_frame=start_frame,
        temporal_interval_mode=temporal_interval_mode,
        fps=fps,
        device=pipe.model.tensor_kwargs["device"],
        mask_fill_value=mask_fill_value,
    )
    decode_method = "erp_0_180_blended" if method.blended_decode else "single"
    (output_dir / "cosmos3_inference_input.json").write_text(
        json.dumps(
            {
                "contract": [
                    "masked_first_rgb",
                    "full_pcd_control",
                    "generated_full_rgb",
                    "prompt",
                ],
                "mask_role": "first_rgb_conditioning_and_frame0_hardlock",
                "rgb_video_path": str(sample.rgb_path.resolve()),
                "first_frame_path": str(first_frame_path.resolve()),
                "pcd_path": str(sample.pcd_path.resolve()),
                "guided_generation_mask_path": str(mask_path.resolve()) if mask_path else None,
                "prompt_path": str(sample.prompt_path.resolve()),
                "height": prepared.height,
                "width": prepared.width,
                "num_frames": int(prepared.pcd_frames.shape[0]),
                "fps": prepared.fps,
                "method": method.name,
                "hardlock_first_frame": method.hardlock_first_frame,
                "hardlock_erode_px": hardlock_erode_px,
                "pcd_mode": "full",
                "decode_method": decode_method,
                "use_ema_weights": use_ema_weights,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if dump_inputs:
        dump_prepared_inputs(
            prepared,
            output_dir=output_dir / "model_inputs",
            prompt=prompt,
            first_frame_path=first_frame_path,
            pcd_path=sample.pcd_path,
            prompt_path=sample.prompt_path,
            mask_path=mask_path,
        )

    sampler = pipe.model.fixed_step_sampler if pipe.model.config.fixed_step_sampler_config is not None else None
    effective_guidance = 1.0 if sampler is not None else guidance
    generated = pipe.model.generate_samples_from_batch(
        prepared.batch,
        sampler=sampler,
        guidance=effective_guidance,
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
    decoded = decode_erp_blended(pipe.model.decode, latent) if method.blended_decode else pipe.model.decode(latent)
    decoded = ((1.0 + decoded) / 2.0).clamp(0.0, 1.0)
    if method.hardlock_first_frame:
        decoded = hardlock_first_frame(
            decoded,
            rgb_first_frame=prepared.first_frame,
            keep_mask=prepared.keep_mask,
            erode_px=hardlock_erode_px,
        )
    generated_frames = int(decoded.shape[2])
    expected_frames = int(prepared.pcd_frames.shape[0])
    if generated_frames != expected_frames:
        raise RuntimeError(f"Cosmos3 decoded {generated_frames} frames, expected {expected_frames}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_img_or_video(decoded[0].cpu(), str(output_path.with_suffix("")), fps=prepared.fps, quality=10)
    (output_dir / "sample_outputs.json").write_text(
        json.dumps(
            {
                "files": [str(output_path)],
                "method": method.name,
                "num_frames": generated_frames,
                "num_steps": num_steps,
                "guidance_requested": guidance,
                "guidance_effective": effective_guidance,
                "shift": shift,
                "seed": seed,
                "pcd_mode": "full",
                "decode_method": decode_method,
                "hardlock_first_frame": method.hardlock_first_frame,
                "hardlock_erode_px": hardlock_erode_px,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del generated, latent, decoded, prepared
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def sample_to_dict(sample: PwPNanoInferenceSample) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(sample).items()
    }
