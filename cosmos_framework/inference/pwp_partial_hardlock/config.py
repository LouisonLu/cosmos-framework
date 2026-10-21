# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Request schema for Cosmos3 PwP partial-hardlock, no-video-path, blended-decode inference.

A request is the Cosmos3 counterpart of a Cosmos2.5
``InferenceArgumentsNoVideoPathBlendedDecode`` JSON: there is no RGB ``video_path``;
the model sees only the masked RGB frame 0, the full PCD control video, and the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pydantic

METHOD_NAME = "cosmos3_partial_hardlock_no_video_path_blended_decode"


class PartialHardlockRequest(pydantic.BaseModel):
    """One inference sample. Relative paths in a request file resolve against that file's directory."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str
    """Output stem: the video is written to ``<output_dir>/<name>.mp4``."""
    prompt: str = pydantic.Field(min_length=1)
    image_context_path: Path
    """RGB frame 0 (PNG/JPG). The guided-generation mask is applied to it before it reaches the model."""
    pcd_path: Path
    """Full, unmasked PCD control video; it also sets the clip length and default fps."""
    guided_generation_mask: Path | None = None
    """Keep mask (PNG/JPG, MP4 frame 0, NPZ ``arr_0[0]``); white is kept and hardlocked. None keeps all of frame 0."""
    guided_generation_mask_erode_px: int = pydantic.Field(default=8, ge=0)
    """Erode the white region before the latent hardlock only; the model's masked frame 0 is not eroded."""
    blended_decoding: bool = True
    """Decode at 0 and 180 degrees and blend them to remove the ERP seam."""
    pixel_composite_first_frame: bool = False
    """After decode, also paste the eroded white region of frame 0 from the source pixels."""
    seed: int = 0
    guidance: float = 6.0
    num_steps: int = pydantic.Field(default=35, gt=0)
    shift: float = 5.0
    width: int = pydantic.Field(default=1024, gt=0)
    height: int = pydantic.Field(default=512, gt=0)
    num_frames: int = pydantic.Field(default=93, gt=0)
    """Frames read from the PCD video, trimmed to 1 + 4k for the VAE."""
    fps: float | None = None
    """Conditioning fps; defaults to the PCD video's fps."""
    mask_fill_value: int = pydantic.Field(default=0, ge=0, le=255)

    def resolve_paths(self, base_dir: Path) -> PartialHardlockRequest:
        def resolve(path: Path | None) -> Path | None:
            if path is None:
                return None
            path = path.expanduser()
            return path if path.is_absolute() else (base_dir / path).resolve()

        return self.model_copy(
            update={
                "image_context_path": resolve(self.image_context_path),
                "pcd_path": resolve(self.pcd_path),
                "guided_generation_mask": resolve(self.guided_generation_mask),
            }
        )


def load_requests(path: str | Path) -> list[PartialHardlockRequest]:
    """Load a ``.json`` (one object or a list) or ``.jsonl`` (one object per line) request file."""
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        payloads: list[Any] = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(text)
        payloads = payload if isinstance(payload, list) else [payload]
    else:
        raise ValueError(f"Request file must be .json or .jsonl: {path}")
    requests = [PartialHardlockRequest.model_validate(item).resolve_paths(path.parent) for item in payloads]
    names = [request.name for request in requests]
    if len(set(names)) != len(names):
        raise ValueError(f"Request names must be unique: {path}")
    return requests
