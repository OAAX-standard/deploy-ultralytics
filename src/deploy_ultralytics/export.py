import json
import shutil
from pathlib import Path

from ultralytics import YOLO

OPSET = 15
DEFAULT_IMGSZ = 640


class ExportError(Exception):
    pass


def export_model(model_name: str, output_dir: str | Path, imgsz: int = DEFAULT_IMGSZ) -> tuple[Path, Path]:
    """Export an Ultralytics detection model to ONNX and write its model manifest.

    Returns the paths of the exported model and the manifest.
    """
    model = YOLO(model_name)
    if model.task != "detect":
        raise ExportError(f"{model_name} has task '{model.task}'; export supports only 'detect' models")

    produced = Path(
        model.export(
            format="onnx",
            opset=OPSET,
            dynamic=False,
            simplify=True,
            imgsz=imgsz,
            nms=False,
        )
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_path = output_dir / produced.name
    shutil.move(produced, exported_path)

    manifest_path = exported_path.with_suffix(".manifest.json")
    manifest = {
        "input_name": _input_name(exported_path),
        "input_resolution": [imgsz, imgsz],
        "class_names": [model.names[i] for i in sorted(model.names)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return exported_path, manifest_path


def _input_name(onnx_path: Path) -> str:
    import onnx

    return onnx.load(onnx_path).graph.input[0].name
