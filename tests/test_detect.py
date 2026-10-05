import json
import os

from helpers import TEST_IMAGE, assert_matches_reference_detections, run


def detect(tmp_path, runtime, exported_model, out):
    return run(
        tmp_path,
        "detect",
        "--target", "cpu",
        "--runtime", str(runtime),
        "--model", str(exported_model),
        "--output", str(out),
        str(TEST_IMAGE),
    )


def test_detect_finds_every_reference_detection(tmp_path, runtime, exported_model):
    out = tmp_path / "out"
    result = detect(tmp_path, runtime, exported_model, out)
    assert result.returncode == 0, result.stderr

    detections = json.loads((out / "test-image.detections.json").read_text())
    assert_matches_reference_detections(detections, "cpu")


def test_detect_writes_the_annotated_image_and_the_json_where_asked(
    tmp_path, runtime, exported_model
):
    out = tmp_path / "somewhere" / "else"
    result = detect(tmp_path, runtime, exported_model, out)
    assert result.returncode == 0, result.stderr
    assert (out / "test-image.annotated.jpg").stat().st_size > 0
    detections = json.loads((out / "test-image.detections.json").read_text())
    assert detections
    assert sorted(detections[0]) == ["box", "class", "score"]


def test_runtime_can_come_from_the_environment(tmp_path, runtime, exported_model):
    result = run(
        tmp_path,
        "detect", "--target", "cpu", "--model", str(exported_model),
        "--output", str(tmp_path / "out"), str(TEST_IMAGE),
        env={**os.environ, "DEPLOY_ULTRALYTICS_RUNTIME": str(runtime)},
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


def test_detect_of_a_model_without_its_manifest_says_to_export_it_first(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    runtime = tmp_path / "libRuntimeLibrary.so"
    runtime.write_bytes(b"")

    result = run(
        tmp_path, "detect", "--target", "cpu", "--runtime", str(runtime), "--model", str(model),
        str(TEST_IMAGE),
    )

    assert result.returncode != 0
    assert "model.manifest.json" in result.stderr and "export" in result.stderr
    assert "Traceback" not in result.stderr
