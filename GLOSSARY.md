# deploy-ultralytics

Examples of running Ultralytics models on several AI accelerators through OAAX, showing that one
inference program serves every accelerator and only the conversion step differs.

## Language

**Target**:
An accelerator the model is deployed on, described by the data needed to convert and feed a model
for it (toolchain, input layout, calibration). One directory per target.
_Avoid_: Vendor, backend, device, accelerator (when meaning the directory or its config)

**Toolchain**:
The OAAX program that converts an ONNX model into the form a target's runtime loads. It takes
either the exported model alone or a bundle, as its target requires.
_Avoid_: Compiler, converter

**Bundle**:
A zip archive given to a toolchain in place of the bare exported model: the exported model plus
the target's fixed toolchain settings (e.g. Intel's choice to keep FP32 inputs). Built anew for
every conversion.
_Avoid_: Archive, package, model zip

**Runtime**:
The OAAX shared library that loads a converted model and runs inference on a target.
_Avoid_: Runtime library, backend

**Converted model**:
The toolchain's output for one model and one target (e.g. ONNX for CPU, a zip of OpenVINO IR for
Intel, `.dxnn` for DeepX).
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

**Walkthrough**:
The section of a target's README that lists the commands which take a model from Ultralytics to
boxes on an image, in order. CI runs it as written, so it is the proof that the documented
commands work.
_Avoid_: Tutorial, quickstart, example commands

**Target config**:
The data file in a target's directory. It holds what conversion changes about the input (element
type, layout, normalization), the toolchain image reference, whether the toolchain takes the
exported model or a bundle, where the runtime and toolchain come from, and the IoU threshold for
matching reference detections. The input description defaults to the exported model's own input
and the threshold to 0.9.
_Avoid_: Model config, preprocessing config
