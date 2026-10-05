import json

from helpers import TEST_IMAGE, assert_matches_reference_detections, run


def convert(tmp_path, model, out):
    return run(tmp_path, "convert", "--target", "cpu", "--output", str(out), str(model))


def test_convert_writes_the_converted_model_and_its_manifest(
    tmp_path, exported_model, toolchain_image
):
    out = tmp_path / "converted"
    result = convert(tmp_path, exported_model, out)

    assert result.returncode == 0, result.stderr
    assert (out / "yolov8n.cpu.onnx").stat().st_size > 0
    exported_manifest = exported_model.with_suffix(".manifest.json").read_text()
    assert (out / "yolov8n.cpu.manifest.json").read_text() == exported_manifest
    assert str(out / "yolov8n.cpu.onnx") in result.stdout


def test_detect_accepts_the_converted_model_and_matches_the_reference_detections(
    tmp_path, exported_model, toolchain_image, runtime
):
    converted = tmp_path / "converted"
    assert convert(tmp_path, exported_model, converted).returncode == 0

    result = run(
        tmp_path,
        "detect", "--target", "cpu", "--runtime", str(runtime),
        "--model", str(converted / "yolov8n.cpu.onnx"),
        "--output", str(tmp_path / "out"), str(TEST_IMAGE),
    )

    assert result.returncode == 0, result.stderr
    detections = json.loads((tmp_path / "out" / "test-image.detections.json").read_text())
    assert_matches_reference_detections(detections, "cpu")


def test_convert_of_a_file_that_is_not_a_model_reports_the_toolchain_failure(
    tmp_path, toolchain_image
):
    not_a_model = tmp_path / "broken.onnx"
    not_a_model.write_text("not an onnx file")
    not_a_model.with_suffix(".manifest.json").write_text("{}")

    result = convert(tmp_path, not_a_model, tmp_path / "converted")

    assert result.returncode != 0
    assert "toolchain" in result.stderr


def test_convert_of_a_model_without_its_manifest_says_to_export_it_first(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")

    result = convert(tmp_path, model, tmp_path / "converted")

    assert result.returncode != 0
    assert "model.manifest.json" in result.stderr and "export" in result.stderr
