import argparse
import sys

_SUBCOMMANDS = {
    "export": "Export an Ultralytics detection model to ONNX and write its model manifest",
    "fetch": "Download the runtime and toolchain for a target",
    "convert": "Convert an exported model for a target with the target's toolchain",
    "detect": "Run detection on an image with a target's runtime",
}


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command is None:
        _build_parser().print_help()
        return 2
    if args.command == "export":
        return _export(args)
    print(f"deploy-ultralytics {args.command}: not implemented yet", file=sys.stderr)
    return 1


def _export(args):
    from deploy_ultralytics.export import ExportError, export_model

    try:
        model_path, manifest_path = export_model(args.model, args.output, args.imgsz)
    except ExportError as error:
        print(f"deploy-ultralytics export: {error}", file=sys.stderr)
        return 1
    print(f"exported model: {model_path}")
    print(f"model manifest: {manifest_path}")
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="deploy-ultralytics",
        description="Run Ultralytics YOLO models on several accelerators through OAAX.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{export,fetch,convert,detect}")
    for name, help_text in _SUBCOMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text, description=help_text)
        if name == "export":
            _add_export_arguments(subparser)
    return parser


def _add_export_arguments(subparser):
    subparser.add_argument("model", help="Ultralytics detection model name, e.g. yolov8n")
    subparser.add_argument("--imgsz", type=int, default=640, help="input resolution (default: 640)")
    subparser.add_argument("--output", default=".", help="output directory (default: current)")
