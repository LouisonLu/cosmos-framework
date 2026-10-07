#!/usr/bin/env python3
"""Create a PwP Stage1 JSONL manifest from legacy RGB/PCD/prompt folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--pcd-dir", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--rgb-suffix", default=".mp4")
    parser.add_argument("--pcd-suffix", default="_pcd.mp4")
    parser.add_argument("--prompt-suffix", default="_prompt.json")
    parser.add_argument("--mask-suffix", default="_mask.png")
    args = parser.parse_args()

    rows = []
    for rgb_path in sorted(args.rgb_dir.glob(f"*{args.rgb_suffix}")):
        stem = rgb_path.name[: -len(args.rgb_suffix)] if args.rgb_suffix else rgb_path.stem
        pcd = args.pcd_dir / f"{stem}{args.pcd_suffix}"
        prompt = args.prompt_dir / f"{stem}{args.prompt_suffix}"
        mask = args.mask_dir / f"{stem}{args.mask_suffix}" if args.mask_dir else None
        missing = [str(path) for path in (pcd, prompt) if not path.is_file()]
        if mask is not None and not mask.is_file():
            missing.append(str(mask))
        if missing:
            raise FileNotFoundError(f"{rgb_path.name}: missing {', '.join(missing)}")
        row = {
            "uuid": stem,
            "rgb_path": str(rgb_path.resolve()),
            "pcd_path": str(pcd.resolve()),
            "prompt_path": str(prompt.resolve()),
        }
        if mask is not None:
            row["mask_path"] = str(mask.resolve())
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No RGB videos matched *{args.rgb_suffix} under {args.rgb_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"PASS: wrote {len(rows)} Stage1 manifest records to {args.output}")


if __name__ == "__main__":
    main()
