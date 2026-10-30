import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

TARGETS_DIR = Path(__file__).parent / "targets"

# What the exported model itself takes as input, and the threshold for matching reference detections.
DEFAULT_ELEMENT_TYPE = "float32"
DEFAULT_LAYOUT = "NCHW"
DEFAULT_NORMALIZATION = {"mean": [0, 0, 0], "std": [255, 255, 255]}
DEFAULT_IOU_THRESHOLD = 0.9

# What a toolchain takes: the bare exported model, or a bundle of it with the target's bundle files.
ToolchainInput = Literal["model", "bundle"]
TOOLCHAIN_INPUTS = get_args(ToolchainInput)


class TargetError(Exception):
    pass


@dataclass(frozen=True)
class TargetConfig:
    """What conversion changes about the model input, the comparison threshold, and where the
    target's runtime and toolchain come from."""

    element_type: str
    layout: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    iou_threshold: float
    store_url: str
    version: str
    vendor: str
    toolchain_image: str
    toolchain_input: ToolchainInput
    bundle_dir: Path
    converted_model_extension: str


def load_target(name: str) -> TargetConfig:
    path = TARGETS_DIR / name / "target.json"
    if not path.is_file():
        known = sorted(p.name for p in TARGETS_DIR.iterdir() if (p / "target.json").is_file())
        raise TargetError(f"unknown target '{name}'; known targets: {', '.join(known)}")
    raw = json.loads(path.read_text())
    normalization = raw.get("normalization", DEFAULT_NORMALIZATION)
    toolchain_input = raw["toolchain"]["input"]
    if toolchain_input not in TOOLCHAIN_INPUTS:
        raise TargetError(
            f"{path}: toolchain input is '{toolchain_input}'; expected {' or '.join(map(repr, TOOLCHAIN_INPUTS))}"
        )
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
        toolchain_input=toolchain_input,
        bundle_dir=path.parent / "bundle",
        converted_model_extension=raw["toolchain"]["converted_model_extension"],
    )
