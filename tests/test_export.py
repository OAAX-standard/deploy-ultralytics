import json
import subprocess
import sys

import onnx


def run(source_models_dir, *args):
    return subprocess.run(
        [sys.executable, "-m", "deploy_ultralytics", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=source_models_dir,
    )


def shape_of(value_info):
    return [d.dim_value for d in value_info.type.tensor_type.shape.dim]


def test_export_writes_the_exported_model_and_manifest(source_models_dir, tmp_path):
    result = run(source_models_dir, "export", "yolov8n", "--output", str(tmp_path))
    assert result.returncode == 0, result.stderr

    model = onnx.load(tmp_path / "yolov8n.onnx")
    manifest = json.loads((tmp_path / "yolov8n.manifest.json").read_text())

    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
    assert shape_of(model.graph.input[0]) == [1, 3, 640, 640]
    batch, channels, anchors = shape_of(model.graph.output[0])
    assert (batch, channels, anchors) == (1, 4 + 80, 8400)

    assert manifest["input_name"] == model.graph.input[0].name
    assert manifest["input_resolution"] == [640, 640]
    assert len(manifest["class_names"]) == 80
    assert manifest["class_names"][0] == "person"
    assert model.opset_import[0].version == 15


def test_imgsz_changes_the_resolution_in_model_and_manifest(source_models_dir, tmp_path):
    result = run(source_models_dir, "export", "yolov8n", "--imgsz", "320", "--output", str(tmp_path))
    assert result.returncode == 0, result.stderr

    model = onnx.load(tmp_path / "yolov8n.onnx")
    manifest = json.loads((tmp_path / "yolov8n.manifest.json").read_text())

    assert shape_of(model.graph.input[0]) == [1, 3, 320, 320]
    assert shape_of(model.graph.output[0]) == [1, 84, 2100]
    assert manifest["input_resolution"] == [320, 320]


def test_export_refuses_a_segmentation_model_and_names_the_task(source_models_dir, tmp_path):
    result = run(source_models_dir, "export", "yolov8n-seg", "--output", str(tmp_path))
    assert result.returncode != 0
    assert "task 'segment'" in result.stderr
    assert list(tmp_path.iterdir()) == []
