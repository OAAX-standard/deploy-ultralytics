import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"
TEST_IMAGE = FIXTURES / "test-image.jpg"


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
