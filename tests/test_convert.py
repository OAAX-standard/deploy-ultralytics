from helpers import CONVERTED_MODEL, TARGET_NAME, run


def convert(tmp_path, model, out):
    return run(tmp_path, "convert", "--target", TARGET_NAME, "--output", str(out), str(model))


def test_convert_writes_the_converted_model_and_its_manifest(tmp_path, exported_model, toolchain_image):
    out = tmp_path / "converted"
    result = convert(tmp_path, exported_model, out)

    assert result.returncode == 0, result.stderr
    converted = out / CONVERTED_MODEL
    assert converted.stat().st_size > 0
    exported_manifest = exported_model.with_suffix(".manifest.json").read_text()
    assert (out / f"yolov8n.{TARGET_NAME}.manifest.json").read_text() == exported_manifest
    assert str(converted) in result.stdout


def test_convert_of_a_file_that_is_not_a_model_reports_the_toolchain_failure(tmp_path, toolchain_image):
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
