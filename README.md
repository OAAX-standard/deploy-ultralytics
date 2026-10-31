# deploy-ultralytics

[![CI](https://github.com/OAAX-standard/deploy-ultralytics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/OAAX-standard/deploy-ultralytics/actions/workflows/ci.yml)

Run an Ultralytics YOLO detection model on several AI accelerators through
[OAAX](https://github.com/OAAX-standard). The inference program is the same for every accelerator;
only the conversion step differs.

> Status: on every push, CI runs the Walkthrough and the test suite of the CPU and Intel targets.
> Both targets are verified; see [Targets](#targets).

## Walkthrough

The README of each target holds the Walkthrough for that target: the commands that take a model
from Ultralytics to boxes on an image. The [Targets](#targets) table links each Walkthrough. CI
runs each Walkthrough as written, in one job for each target. The commands differ between targets
in the target name, the converted model and the runtime. `export` and the exported model stay the
same.

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

## Fetch

```sh
deploy-ultralytics fetch cpu
```

`fetch` downloads a target's runtime archive and toolchain image archive from the OAAX release
artifact store. It extracts the runtime archive into `runtimes/cpu/` (under `--output`, default:
the current directory) and prints the path of `libRuntimeLibrary.so` in it. It loads the toolchain
image into Docker and tags the image with the name in the target's `target.json`. The store URL,
the version and the vendor also come from `target.json`.

To use another store, set `DEPLOY_ULTRALYTICS_ARTIFACTS` to a local directory or a URL laid out
the same way:

```
runtimes/2.0.0/CPU/x86_64/Ubuntu/library.tar.gz
conversion-toolchain/2.0.0/CPU/oaax-cpu-toolchain.tar
```

The runtime archive holds `libRuntimeLibrary.so` and the libraries it loads, at any depth. The
toolchain archive is the output of `docker save` of the toolchain image, under any tag.

## Convert

```sh
deploy-ultralytics convert yolov8n.onnx --target cpu --output converted
```

`convert` runs the target's toolchain image on the exported model and writes the converted model
to the output directory as `yolov8n.cpu.onnx`, with the model manifest copied beside it as
`yolov8n.cpu.manifest.json`; that is where `detect` looks for it. The command needs the toolchain
image in Docker, from `fetch`.

Some toolchains take a bundle, not the bare exported model. If `toolchain.input` in the target's
`target.json` is `bundle`, `convert` puts the exported model and the files of the target's
`bundle/` directory in a zip, and runs the toolchain on that zip.

## Detect

```sh
deploy-ultralytics detect --target cpu --runtime path/to/libRuntimeLibrary.so --model yolov8n.cpu.onnx --output out image.jpg
```

`detect` loads the runtime through [PythonBindings](https://github.com/OAAX-standard/PythonBindings)
(the `oaax` package), loads the converted model, letterboxes the image to the model's input
resolution, applies the target's input description (element type, layout, normalization), runs
inference, decodes the `(1, 4 + num_classes, N)` output and applies non-maximum suppression on the
host. It writes two files into the output directory (default: the current directory):

- `<image>.annotated.jpg`, the image with the boxes drawn.
- `<image>.detections.json`, a list of `{"box": [x1, y1, x2, y2], "class": 0, "score": 0.855}` in
  the pixels of the original image, in the same format as the
  [reference detections](fixtures/README.md).

Before inference, `detect` checks the model input against the tensor that it makes from the target
config and the model manifest. It compares the element type and the shape. If they differ, `detect`
stops with a message that names the difference. `detect` gets the model input from the runtime when
the runtime reports it. Otherwise it reads the input from the converted model, if that model is an
ONNX file. If neither source gives the input, `detect` prints `input check skipped` and runs without
the check.

The model manifest is read from next to the model (`yolov8n.cpu.onnx` -> `yolov8n.cpu.manifest.json`). The
runtime comes from `--runtime`, or from the `DEPLOY_ULTRALYTICS_RUNTIME` environment
variable when the option is absent. Tests that need a runtime are skipped, with a message, when
the variable is not set.

## Targets

| Target | Status  |
| ------ | ------- |
| [CPU](src/deploy_ultralytics/targets/cpu/README.md#walkthrough) | export -> convert -> detect verified in CI |
| [Intel](src/deploy_ultralytics/targets/intel/README.md#walkthrough) | export -> convert -> detect verified in CI |
| NVIDIA | pending |
| DeepX  | pending |

The Intel runtime reports the model input, and `detect` checks against that report. The CPU runtime
does not report it, so `detect` reads the input from the converted model, which is an ONNX file.

## Install

Requires Python 3.10 or newer.

```sh
uv sync
# or
pip install --extra-index-url https://download.pytorch.org/whl/cpu .
```

Nothing here runs torch on a GPU: `export` runs on the CPU, and inference goes through the OAAX
runtimes. `pyproject.toml` therefore takes `torch` and `torchvision` from the PyTorch CPU index,
which spares the install several GB of CUDA wheels. `uv sync` applies that setting. `pip` does not,
so give it the index with `--extra-index-url`. Without it, `pip` installs the CUDA build of torch
from PyPI.

The C++ example this repository started from was dropped; see
[ADR-0001](docs/adr/0001-python-over-legacy-cpp.md).

## Reference detections

The boxes Ultralytics itself predicts for the test image are committed under
[`fixtures/`](fixtures/README.md), with the command that regenerates them. Each target's `detect`
output is judged against them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Ultralytics is called as a dependency only; none of
its code is copied here.
