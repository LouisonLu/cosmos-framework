from pathlib import Path

import pytest

from cosmos_framework.inference.pwp_nano_inference_batch import (
    PwPNanoInferenceSample,
    build_output_filename,
    discover_samples,
    resolve_inference_method,
    select_sample_range,
)


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


def test_discover_samples_can_require_paired_mask(tmp_path: Path):
    _write(tmp_path / "rgb_videos" / "scene.mp4")
    _write(tmp_path / "pcd_videos" / "scene_pcd.mp4")
    _write(tmp_path / "prompts" / "scene_prompt.json", b'{"caption":"a car"}')

    with pytest.raises(FileNotFoundError, match="Missing per-sample mask"):
        discover_samples(tmp_path, require_mask=True)


def test_output_filename_contains_stem_checkpoint_and_method():
    assert (
        build_output_filename(
            "scene id",
            "/models/iter_000002000_bf16",
            "cosmos3_no_video_path_hardlock_blended_decode",
        )
        == "scene-id_iter_000002000_bf16_cosmos3_no_video_path_hardlock_blended_decode.mp4"
    )


def test_output_filename_accepts_explicit_metadata_tag():
    assert build_output_filename(
        "ood__nuscenes__scene-0765",
        "iter_000002000_bf16",
        "cosmos3_no_video_path_hardlock_blended_decode",
        output_tag="cosmos3_iter_000002000_bf16_hardlock_steps35_erode8",
    ) == (
        "ood__nuscenes__scene-0765__"
        "cosmos3_iter_000002000_bf16_hardlock_steps35_erode8.mp4"
    )


def test_select_sample_range_applies_limit_after_start_index():
    samples = [
        PwPNanoInferenceSample(str(i), Path("rgb"), Path("pcd"), Path("prompt"), None)
        for i in range(50)
    ]

    assert [sample.stem for sample in select_sample_range(samples, 0, 10)] == [
        str(i) for i in range(10)
    ]
    assert [sample.stem for sample in select_sample_range(samples, 10, 40)] == [
        str(i) for i in range(10, 50)
    ]


def test_method_names_and_cosmos25_aliases_resolve():
    blended = "cosmos3_no_video_path_hardlock_blended_decode"
    assert resolve_inference_method(blended).name == blended
    assert resolve_inference_method("cosmos3_no_video_path_blended_decode.py").name == blended
    assert resolve_inference_method("inference_partial_hardlock_no_video_path_blended_decode.py").name == blended
    assert resolve_inference_method("examples/inference_partial_hardlock_no_video_path.py").name == (
        "cosmos3_no_video_path_hardlock"
    )

    with pytest.raises(ValueError, match="Unsupported inference method"):
        resolve_inference_method("stage1")


def test_methods_select_decode_and_hardlock():
    blended = resolve_inference_method("cosmos3_no_video_path_hardlock_blended_decode")
    assert blended.blended_decode and blended.hardlock_first_frame
    single = resolve_inference_method("cosmos3_no_video_path_hardlock")
    assert not single.blended_decode and single.hardlock_first_frame
    raw = resolve_inference_method("cosmos3_no_video_path")
    assert not raw.blended_decode and not raw.hardlock_first_frame


def test_discover_samples_finds_cosmos25_mask_dir_and_legacy_fallback(tmp_path: Path):
    for name in ("a", "b"):
        _write(tmp_path / name / "rgb_videos" / "scene.mp4")
        _write(tmp_path / name / "pcd_videos" / "scene_pcd.mp4")
        _write(tmp_path / name / "prompts" / "scene_prompt.json", b'{"caption":"a car"}')
    _write(tmp_path / "a" / "mask" / "scene_mask.mp4")
    _write(tmp_path / "b" / "mask_videos" / "scene_mask.mp4")

    assert discover_samples(tmp_path / "a")[0].mask_path == (tmp_path / "a" / "mask" / "scene_mask.mp4").resolve()
    assert discover_samples(tmp_path / "b")[0].mask_path == (
        tmp_path / "b" / "mask_videos" / "scene_mask.mp4"
    ).resolve()
