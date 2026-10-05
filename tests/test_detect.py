import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"
TEST_IMAGE = FIXTURES / "test-image.jpg"
IOU_THRESHOLD = 0.9  # the cpu target's threshold


@pytest.fixture(scope="session")
def runtime_library():
    path = os.environ.get("DEPLOY_ULTRALYTICS_RUNTIME")
    if not path:
        pytest.skip("DEPLOY_ULTRALYTICS_RUNTIME is not set: no runtime library to run detect with")
    assert Path(path).is_file(), f"DEPLOY_ULTRALYTICS_RUNTIME={path} is not a file"
    return Path(path)


@pytest.fixture(scope="session")
def exported_model(weights_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("exported")
    result = run(weights_dir, "export", "yolov8n", "--output", str(out))
    assert result.returncode == 0, result.stderr
    return out / "yolov8n.onnx"


def run(cwd, *args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "deploy_ultralytics", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
    )


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    intersection = max(width, 0) * max(height, 0)
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return intersection / (area(a) + area(b) - intersection)


def detect(tmp_path, runtime_library, exported_model, out):
    return run(
        tmp_path,
        "detect",
        "--target", "cpu",
        "--runtime", str(runtime_library),
        "--model", str(exported_model),
        "--output", str(out),
        str(TEST_IMAGE),
    )


def test_detect_finds_every_reference_detection(tmp_path, runtime_library, exported_model):
    out = tmp_path / "out"
    result = detect(tmp_path, runtime_library, exported_model, out)
    assert result.returncode == 0, result.stderr

    detections = json.loads((out / "test-image.detections.json").read_text())
    reference = json.loads((FIXTURES / "test-image.yolov8n.json").read_text())["detections"]
    for expected in reference:
        best = max(
            (iou(expected["box"], d["box"]) for d in detections if d["class"] == expected["class"]),
            default=0,
        )
        assert best >= IOU_THRESHOLD, f"no class {expected['class']} detection matches {expected}"


def test_detect_writes_the_annotated_image_and_the_json_where_asked(
    tmp_path, runtime_library, exported_model
):
    out = tmp_path / "somewhere" / "else"
    result = detect(tmp_path, runtime_library, exported_model, out)
    assert result.returncode == 0, result.stderr
    assert (out / "test-image.annotated.jpg").stat().st_size > 0
    detections = json.loads((out / "test-image.detections.json").read_text())
    assert detections
    assert sorted(detections[0]) == ["box", "class", "score"]


def test_runtime_library_can_come_from_the_environment(tmp_path, runtime_library, exported_model):
    result = run(
        tmp_path,
        "detect", "--target", "cpu", "--model", str(exported_model),
        "--output", str(tmp_path / "out"), str(TEST_IMAGE),
        env={**os.environ, "DEPLOY_ULTRALYTICS_RUNTIME": str(runtime_library)},
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "test-image.detections.json").exists()


def test_detect_without_a_runtime_says_how_to_provide_one(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "DEPLOY_ULTRALYTICS_RUNTIME"}
    result = run(
        tmp_path,
        "detect", "--target", "cpu", "--model", "model.onnx", str(TEST_IMAGE),
        env=env,
    )
    assert result.returncode != 0
    assert "--runtime" in result.stderr and "DEPLOY_ULTRALYTICS_RUNTIME" in result.stderr
