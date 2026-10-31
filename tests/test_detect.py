import json
import os
import shutil

import pytest
from helpers import TARGET_NAME, TEST_IMAGE, assert_matches_reference_detections, run

from deploy_ultralytics import target
from deploy_ultralytics.cli import main
from deploy_ultralytics.detect import RUNTIME_ENV_VAR
from deploy_ultralytics.target import TARGETS_DIR


def detect(tmp_path, runtime, converted_model, out):
    return run(
        tmp_path,
        "detect",
        "--target",
        TARGET_NAME,
        "--runtime",
        str(runtime),
        "--model",
        str(converted_model),
        "--output",
        str(out),
        str(TEST_IMAGE),
    )


def test_detect_finds_every_reference_detection(tmp_path, runtime, converted_model):
    out = tmp_path / "out"
    result = detect(tmp_path, runtime, converted_model, out)
    assert result.returncode == 0, result.stderr

    detections = json.loads((out / "test-image.detections.json").read_text())
    assert_matches_reference_detections(detections)


def test_detect_writes_the_annotated_image_and_the_json_where_asked(tmp_path, runtime, converted_model):
    out = tmp_path / "somewhere" / "else"
    result = detect(tmp_path, runtime, converted_model, out)
    assert result.returncode == 0, result.stderr
    assert (out / "test-image.annotated.jpg").stat().st_size > 0
    detections = json.loads((out / "test-image.detections.json").read_text())
    assert detections
    assert sorted(detections[0]) == ["box", "class", "score"]


def test_runtime_can_come_from_the_environment(tmp_path, runtime, converted_model):
    result = run(
        tmp_path,
        "detect",
        "--target",
        TARGET_NAME,
        "--model",
        str(converted_model),
        "--output",
        str(tmp_path / "out"),
        str(TEST_IMAGE),
        env={**os.environ, RUNTIME_ENV_VAR: str(runtime)},
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "test-image.detections.json").exists()


def test_detect_without_a_runtime_says_how_to_provide_one(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != RUNTIME_ENV_VAR}
    result = run(
        tmp_path,
        "detect",
        "--target",
        TARGET_NAME,
        "--model",
        "model.onnx",
        str(TEST_IMAGE),
        env=env,
    )
    assert result.returncode != 0
    assert "--runtime" in result.stderr and RUNTIME_ENV_VAR in result.stderr


def test_detect_of_a_model_without_its_manifest_says_to_export_it_first(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    runtime = tmp_path / "libRuntimeLibrary.so"
    runtime.write_bytes(b"")

    result = run(
        tmp_path,
        "detect",
        "--target",
        TARGET_NAME,
        "--runtime",
        str(runtime),
        "--model",
        str(model),
        str(TEST_IMAGE),
    )

    assert result.returncode != 0
    assert "model.manifest.json" in result.stderr and "export" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("override", "model_says", "config_says"),
    [
        ({"element_type": "uint8"}, "float32", "uint8"),
        ({"layout": "NHWC"}, "[1, 3, 640, 640]", "[1, 640, 640, 3]"),
    ],
)
def test_detect_refuses_a_target_config_that_disagrees_with_the_model_input(
    tmp_path, monkeypatch, capsys, runtime, converted_model, override, model_says, config_says
):
    targets_dir = tmp_path / "targets"
    shutil.copytree(TARGETS_DIR / TARGET_NAME, targets_dir / TARGET_NAME)
    config_path = targets_dir / TARGET_NAME / "target.json"
    config_path.write_text(json.dumps({**json.loads(config_path.read_text()), **override}))
    monkeypatch.setattr(target, "TARGETS_DIR", targets_dir)
    out = tmp_path / "out"

    returncode = main(
        ["detect", "--target", TARGET_NAME, "--runtime", str(runtime), "--model", str(converted_model)]
        + ["--output", str(out), str(TEST_IMAGE)]
    )

    stderr = capsys.readouterr().err
    assert returncode != 0
    assert model_says in stderr and config_says in stderr
    assert not out.exists()
