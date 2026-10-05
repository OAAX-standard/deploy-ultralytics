# deploy-ultralytics

Examples of running Ultralytics models on several AI accelerators through OAAX, showing that one
inference program serves every accelerator and only the conversion step differs.

## Language

**Target**:
An accelerator the model is deployed on, described by the data needed to convert and feed a model
for it (toolchain, input layout, calibration). One directory per target.
_Avoid_: Vendor, backend, device, accelerator (when meaning the directory or its config)

**Toolchain**:
The OAAX program that converts an ONNX model into the form a target's runtime loads.
_Avoid_: Compiler, converter

**Runtime**:
The OAAX shared library that loads a converted model and runs inference on a target.
_Avoid_: Runtime library, backend

**Converted model**:
The toolchain's output for one model and one target (e.g. ONNX for CPU, OpenVINO IR for Intel,
`.dxnn` for DeepX).
_Avoid_: Compiled model, artifact

**Source model**:
The Ultralytics checkpoint (e.g. `yolov8n.pt`) before export.
_Avoid_: Weights, checkpoint

**Exported model**:
The ONNX file produced from a source model, identical for every target.

**Model manifest**:
The description written next to an exported model: what the model itself fixes (input name,
resolution, class names). Independent of any target.
_Avoid_: Model config, model JSON

**Reference detections**:
The boxes and classes Ultralytics itself produces for a source model on the test image, stored as
a fixture. Every target's `detect` output is judged against them.
_Avoid_: Ground truth, golden output, baseline

**Target config**:
The data file in a target's directory. It holds what conversion changes about the input (element
type, layout, normalization), the toolchain image reference, where the runtime and toolchain come
from, and the IoU threshold for matching reference detections. The input description defaults to
the exported model's own input and the threshold to 0.9.
_Avoid_: Model config, preprocessing config
