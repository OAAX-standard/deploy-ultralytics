# deploy-ultralytics

Run an Ultralytics YOLO detection model on several AI accelerators through
[OAAX](https://github.com/OAAX-standard). The inference program is the same for every accelerator;
only the conversion step differs.

> Status: work in progress. The commands below show the intended flow; none of the targets is
> verified yet.

## Walkthrough

```sh
deploy-ultralytics export yolov8n.pt
deploy-ultralytics convert yolov8n.onnx --target cpu
deploy-ultralytics detect  --target cpu --model yolov8n.onnx bus.jpg
```

To run on another accelerator, change only `--target`. `fetch` downloads the runtime and toolchain
a target needs from the OAAX release artifacts.

| Subcommand | What it does                                                          |
| ---------- | --------------------------------------------------------------------- |
| `export`   | Export an Ultralytics detection model to ONNX and write its manifest  |
| `fetch`    | Download the runtime and toolchain for a target                       |
| `convert`  | Convert the exported model with the target's toolchain                |
| `detect`   | Run detection on an image; writes an annotated image and a JSON list  |

## Export

```sh
deploy-ultralytics export yolov8n --output exported
```

`export` takes the name of an Ultralytics detection model (downloaded on first use into the
current directory) and writes two files into the output directory (default: the current
directory):

- `yolov8n.onnx`, the exported model: ONNX opset 15, static shapes, simplified, one input of shape
  `(1, 3, H, W)` and one output of shape `(1, 4 + num_classes, N)`. Non-maximum suppression is not
  part of the graph; `detect` applies it on the host.
- `yolov8n.manifest.json`, the model manifest: the input tensor name, the input resolution and the
  class names.

The resolution is 640 by default; `--imgsz 320` changes it in both files. A model whose task is not
detection, such as `yolov8n-seg`, is refused with a message naming its task. The `ultralytics`
package is called as a dependency; none of its code is copied.

## Detect

```sh
deploy-ultralytics detect --target cpu --runtime path/to/libRuntimeLibrary.so     --model exported/yolov8n.onnx --output out image.jpg
```

`detect` loads the runtime through [PythonBindings](https://github.com/OAAX-standard/PythonBindings),
loads the converted model, letterboxes the image to the model's input resolution, applies the
target's input description (element type, layout, normalization), runs inference, decodes the
`(1, 4 + num_classes, N)` output and applies non-maximum suppression on the host. It writes two
files into the output directory (default: the current directory):

- `<image>.annotated.jpg`, the image with the boxes drawn.
- `<image>.detections.json`, a list of `{"box": [x1, y1, x2, y2], "class": 0, "score": 0.855}` in
  the pixels of the original image, in the same format as the
  [reference detections](fixtures/README.md).

The model manifest is read from next to the model (`yolov8n.onnx` -> `yolov8n.manifest.json`). The
runtime library comes from `--runtime`, or from the `DEPLOY_ULTRALYTICS_RUNTIME` environment
variable when the option is absent. Tests that need a runtime are skipped, with a message, when
the variable is not set.

No OAAX 2.0 runtime is published yet. Build the CPU one from the `oaax-v2` branch of
[reference-implementation](https://github.com/OAAX-standard/reference-implementation)
(`runtime-library/build-runtimes.sh X86_64`, Linux only) and point the variable at the resulting
`libRuntimeLibrary.so`.

## Targets

| Target | Status  |
| ------ | ------- |
| CPU    | export -> detect runs; detect's input check is pending (see below) |
| Intel  | pending |
| NVIDIA | pending |
| DeepX  | pending |

`detect` does not yet check the runtime's reported input shape and element type against the
manifest and the target config: the PythonBindings the repository pins read that information from a
`models` entry of the runtime's info JSON, which no OAAX 2.0 runtime emits. Until that is fixed
upstream, CPU stays unverified.

## Install

Requires Python 3.10 or newer and access to the
[PythonBindings](https://github.com/OAAX-standard/PythonBindings) repository.

```sh
pip install .
# or
uv sync
```

The C++ example this repository started from was dropped; see
[ADR-0001](docs/adr/0001-python-over-legacy-cpp.md).

## Reference detections

The boxes Ultralytics itself predicts for the test image are committed under
[`fixtures/`](fixtures/README.md), with the command that regenerates them. Each target's `detect`
output is judged against them.

## License

Apache-2.0. Ultralytics is called as a dependency only; none of its code is copied here.
