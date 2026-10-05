import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"


def regenerate(tmp_path, model="yolov8n.pt"):
    out = tmp_path / "reference.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy_ultralytics.reference",
            model,
            str(FIXTURES / "test-image.jpg"),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,  # Ultralytics downloads the source model into the working directory
    )
    return result, out


def test_committed_reference_detections_match_a_fresh_regeneration(tmp_path):
    result, out = regenerate(tmp_path)
    assert result.returncode == 0, result.stderr
    committed = json.loads((FIXTURES / "test-image.yolov8n.json").read_text())
    assert json.loads(out.read_text()) == committed


def test_reference_detections_list_box_class_and_score(tmp_path):
    result, out = regenerate(tmp_path)
    assert result.returncode == 0, result.stderr
    detections = json.loads(out.read_text())["detections"]
    assert detections
    for detection in detections:
        assert sorted(detection) == ["box", "class", "score"]
        assert len(detection["box"]) == 4
