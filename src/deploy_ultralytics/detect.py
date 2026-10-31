import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from deploy_ultralytics.postprocess import decode
from deploy_ultralytics.preprocess import letterbox, to_input_tensor
from deploy_ultralytics.target import load_target

RUNTIME_ENV_VAR = "DEPLOY_ULTRALYTICS_RUNTIME"


class DetectError(Exception):
    pass


def detect_image(
    target_name: str,
    runtime: str | None,
    model_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Run a converted model on an image through the target's runtime.

    Writes `<image stem>.annotated.jpg` and `<image stem>.detections.json` into `output_dir`
    and returns their paths.
    """
    library = _runtime_path(runtime)
    target = load_target(target_name)
    model_path = Path(model_path)
    manifest_path = model_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise DetectError(f"no model manifest {manifest_path} next to the model: run export to write it")
    manifest = json.loads(manifest_path.read_text())
    image = cv2.imread(str(image_path))
    if image is None:
        raise DetectError(f"cannot read image {image_path}")

    padded, box_mapping = letterbox(image, manifest["input_resolution"])
    tensor = to_input_tensor(padded, target)

    from oaax import Runtime
    from oaax.onnx_introspect import model_io

    with Runtime(library) as loaded:
        loaded.init({})
        model_id = loaded.load_model(model_path)
        io, reason = model_io(loaded, model_id, model_path)
        if io is None:
            print(f"deploy-ultralytics detect: input check skipped: {reason}", file=sys.stderr)
        else:
            _check_input(io[0], manifest["input_name"], tensor, target_name)
        result = loaded.run(model_id, {manifest["input_name"]: tensor})
    detections = decode(result[0].data, box_mapping, image.shape)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    json_path = output_dir / f"{stem}.detections.json"
    json_path.write_text(json.dumps(detections, indent=2) + "\n")
    annotated_path = output_dir / f"{stem}.annotated.jpg"
    cv2.imwrite(str(annotated_path), _annotate(image, detections, manifest["class_names"]))
    return annotated_path, json_path


def _runtime_path(option: str | None) -> str:
    path = option or os.environ.get(RUNTIME_ENV_VAR)
    if not path:
        raise DetectError(f"no runtime: pass --runtime or set {RUNTIME_ENV_VAR}")
    if not Path(path).is_file():
        raise DetectError(f"runtime {path} does not exist")
    return path


def _check_input(inputs, name: str, tensor: np.ndarray, target_name: str) -> None:
    """Refuse a model whose input differs from the tensor the target config and the manifest give."""
    spec = next((spec for spec in inputs if spec.name == name), None)
    if spec is None:
        raise DetectError(f"the model has no input '{name}' named in its manifest; it has {[s.name for s in inputs]}")
    element_type = spec.dtype.name if spec.dtype is not None else spec.element_type
    if element_type != tensor.dtype.name:
        raise DetectError(
            f"the model's input '{name}' has element type {element_type}; "
            f"target config '{target_name}' gives {tensor.dtype.name}"
        )
    if list(spec.shape) != list(tensor.shape):
        raise DetectError(
            f"the model's input '{name}' has shape {list(spec.shape)}; "
            f"target config '{target_name}' and the manifest give {list(tensor.shape)}"
        )


def _annotate(image: np.ndarray, detections: list[dict], class_names: list[str]) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (round(v) for v in detection["box"])
        label = f"{class_names[detection['class']]} {detection['score']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, label, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return annotated
