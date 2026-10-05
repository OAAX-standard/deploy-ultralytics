import json
from dataclasses import dataclass
from pathlib import Path

TARGETS_DIR = Path(__file__).parent / "targets"


class TargetError(Exception):
    pass


@dataclass(frozen=True)
class TargetConfig:
    """What conversion changes about the model input, plus the comparison threshold."""

    element_type: str
    layout: str
    mean: tuple
    std: tuple
    iou_threshold: float


def load_target(name):
    path = TARGETS_DIR / name / "target.json"
    if not path.is_file():
        known = sorted(p.name for p in TARGETS_DIR.iterdir() if (p / "target.json").is_file())
        raise TargetError(f"unknown target '{name}'; known targets: {', '.join(known)}")
    raw = json.loads(path.read_text())
    return TargetConfig(
        element_type=raw["element_type"],
        layout=raw["layout"],
        mean=tuple(raw["normalization"]["mean"]),
        std=tuple(raw["normalization"]["std"]),
        iou_threshold=raw["iou_threshold"],
    )
