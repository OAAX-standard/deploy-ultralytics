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
deploy-ultralytics detect  yolov8n.onnx --target cpu --image bus.jpg
```

To run on another accelerator, change only `--target`. `fetch` downloads the runtime and toolchain
a target needs from the OAAX release artifacts.

| Subcommand | What it does                                                          |
| ---------- | --------------------------------------------------------------------- |
| `export`   | Export an Ultralytics detection model to ONNX and write its manifest  |
| `fetch`    | Download the runtime and toolchain for a target                       |
| `convert`  | Convert the exported model with the target's toolchain                |
| `detect`   | Run detection on an image; writes an annotated image and a JSON list  |

## Targets

| Target | Status  |
| ------ | ------- |
| CPU    | pending |
| Intel  | pending |
| NVIDIA | pending |
| DeepX  | pending |

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

## License

Apache-2.0. Ultralytics is called as a dependency only; none of its code is copied here.
