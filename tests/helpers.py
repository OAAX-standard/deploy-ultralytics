import json
import os
import subprocess
import sys
from pathlib import Path

from deploy_ultralytics.target import load_target

FIXTURES = Path(__file__).parent.parent / "fixtures"
TEST_IMAGE = FIXTURES / "test-image.jpg"

# The target the tests convert for and detect on; CI runs the suite once per target.
TARGET_ENV_VAR = "DEPLOY_ULTRALYTICS_TARGET"
TARGET_NAME = os.environ.get(TARGET_ENV_VAR, "cpu")
TARGET = load_target(TARGET_NAME)
CONVERTED_MODEL = f"yolov8n.{TARGET_NAME}.{TARGET.converted_model_extension}"


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
    return intersection / (_area(a) + _area(b) - intersection)


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def assert_matches_reference_detections(detections):
    """Every reference detection has a detection of its class overlapping it by the target's IoU threshold."""
    threshold = TARGET.iou_threshold
    reference = json.loads((FIXTURES / "test-image.yolov8n.json").read_text())["detections"]
    for expected in reference:
        best = max(
            (iou(expected["box"], d["box"]) for d in detections if d["class"] == expected["class"]),
            default=0,
        )
        assert best >= threshold, f"no class {expected['class']} detection matches {expected}"
