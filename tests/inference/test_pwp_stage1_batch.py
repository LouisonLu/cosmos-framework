from pathlib import Path

import pytest

from cosmos_framework.inference.pwp_stage1_batch import build_output_filename, discover_samples


def _write(path: Path, content: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_discover_samples_uses_common_mask_and_prompt_suffix(tmp_path: Path):
    _write(tmp_path / "rgb_videos" / "scene.mp4")
    _write(tmp_path / "pcd_videos" / "scene_pcd.mp4")
    _write(tmp_path / "prompts" / "scene_prompt.json", b'{"caption":"a car"}')
    mask = tmp_path / "common.png"
    _write(mask)

    samples = discover_samples(tmp_path, common_mask=mask)

    assert len(samples) == 1
    assert samples[0].stem == "scene"
    assert samples[0].mask_path == mask.resolve()


def test_discover_samples_rejects_missing_pair(tmp_path: Path):
    _write(tmp_path / "rgb_videos" / "scene.mp4")
    _write(tmp_path / "prompts" / "scene_prompt.json", b'{"caption":"a car"}')

    with pytest.raises(FileNotFoundError, match="PCD video"):
        discover_samples(tmp_path)


def test_output_filename_contains_stem_checkpoint_and_method():
    assert (
        build_output_filename(
            "scene id",
            "/models/iter_000002000_bf16",
            "examples/inference_pwp_stage1.py",
        )
        == "scene-id_iter_000002000_bf16_inference_pwp_stage1.mp4"
    )
