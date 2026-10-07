#!/usr/bin/env python3
"""Validate PwP Stage1 manifest paths and RGB/PCD video geometry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def probe(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "v:0", path],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream.get("nb_frames") or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-frames", type=int, default=93)
    parser.add_argument("--expected-width", type=int, default=1024)
    parser.add_argument("--expected-height", type=int, default=512)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("Manifest is empty")
    for index, row in enumerate(rows):
        rgb = Path(row.get("rgb_path", row.get("vision_path", "")))
        pcd = Path(row.get("pcd_path", row.get("control_path", "")))
        if not rgb.is_file() or not pcd.is_file():
            raise FileNotFoundError(f"record {index}: RGB={rgb} PCD={pcd}")
        rgb_info, pcd_info = probe(str(rgb)), probe(str(pcd))
        if rgb_info["frames"] and pcd_info["frames"] and rgb_info["frames"] < args.expected_frames:
            raise ValueError(
                f"record {index}: RGB has {rgb_info['frames']} frames, "
                f"expected >= {args.expected_frames}"
            )
        if pcd_info["frames"] and pcd_info["frames"] < args.expected_frames:
            raise ValueError(
                f"record {index}: PCD has {pcd_info['frames']} frames, "
                f"expected >= {args.expected_frames}"
            )
        if rgb_info["frames"] and pcd_info["frames"] and rgb_info["frames"] != pcd_info["frames"]:
            raise ValueError(f"record {index}: RGB/PCD frame counts differ: {rgb_info} vs {pcd_info}")
        if rgb_info["width"] != args.expected_width or rgb_info["height"] != args.expected_height:
            print(
                f"WARN: record {index} source RGB is {rgb_info['width']}x{rgb_info['height']}; "
                "loader resizes to target"
            )
    print(f"PASS: validated {len(rows)} Stage1 records")


if __name__ == "__main__":
    main()
