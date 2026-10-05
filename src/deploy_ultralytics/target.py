import json
from dataclasses import dataclass
from pathlib import Path

TARGETS_DIR = Path(__file__).parent / "targets"

# What the exported model itself takes as input, and the threshold for matching reference detections.
DEFAULT_ELEMENT_TYPE = "float32"
DEFAULT_LAYOUT = "NCHW"
DEFAULT_NORMALIZATION = {"mean": [0, 0, 0], "std": [255, 255, 255]}
DEFAULT_IOU_THRESHOLD = 0.9


class TargetError(Exception):
    pass


@dataclass(frozen=True)
class TargetConfig:
    """What conversion changes about the model input, the comparison threshold, and where the
    target's runtime and toolchain come from."""

    element_type: str
    layout: str
    mean: tuple
    std: tuple
    iou_threshold: float
    store_url: str
    version: str
    vendor: str
    toolchain_image: str
    converted_model_extension: str


def load_target(name):
    path = TARGETS_DIR / name / "target.json"
    if not path.is_file():
        known = sorted(p.name for p in TARGETS_DIR.iterdir() if (p / "target.json").is_file())
        raise TargetError(f"unknown target '{name}'; known targets: {', '.join(known)}")
    raw = json.loads(path.read_text())
    normalization = raw.get("normalization", DEFAULT_NORMALIZATION)
    return TargetConfig(
        element_type=raw.get("element_type", DEFAULT_ELEMENT_TYPE),
        layout=raw.get("layout", DEFAULT_LAYOUT),
        mean=tuple(normalization["mean"]),
        std=tuple(normalization["std"]),
        iou_threshold=raw.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
        store_url=raw["source"]["store_url"],
        version=raw["source"]["version"],
        vendor=raw["source"]["vendor"],
        toolchain_image=raw["toolchain"]["image"],
        converted_model_extension=raw["toolchain"]["converted_model_extension"],
    )
