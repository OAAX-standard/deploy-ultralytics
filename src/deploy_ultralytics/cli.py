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
    if args.command == "fetch":
        return _fetch(args)
    if args.command == "convert":
        return _convert(args)
    return _detect(args)


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


def _fetch(args):
    from deploy_ultralytics.fetch import FetchError, fetch_target
    from deploy_ultralytics.target import TargetError

    try:
        library = fetch_target(args.target, args.output)
    except (FetchError, TargetError) as error:
        print(f"deploy-ultralytics fetch: {error}", file=sys.stderr)
        return 1
    print(f"runtime library: {library}")
    print(f"toolchain image loaded into Docker; set DEPLOY_ULTRALYTICS_RUNTIME={library}")
    return 0


def _convert(args):
    from deploy_ultralytics.convert import ConvertError, convert_model
    from deploy_ultralytics.target import TargetError

    try:
        model_path, manifest_path = convert_model(args.target, args.model, args.output)
    except (ConvertError, TargetError) as error:
        print(f"deploy-ultralytics convert: {error}", file=sys.stderr)
        return 1
    print(f"converted model: {model_path}")
    print(f"model manifest: {manifest_path}")
    return 0


def _detect(args):
    from deploy_ultralytics.detect import DetectError, detect_image
    from deploy_ultralytics.target import TargetError

    try:
        annotated_path, json_path = detect_image(
            args.target, args.runtime, args.model, args.image, args.output
        )
    except (DetectError, TargetError) as error:
        print(f"deploy-ultralytics detect: {error}", file=sys.stderr)
        return 1
    print(f"annotated image: {annotated_path}")
    print(f"detections: {json_path}")
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
        if name == "fetch":
            _add_fetch_arguments(subparser)
        if name == "convert":
            _add_convert_arguments(subparser)
        if name == "detect":
            _add_detect_arguments(subparser)
    return parser


def _add_export_arguments(subparser):
    subparser.add_argument("model", help="Ultralytics detection model name, e.g. yolov8n")
    subparser.add_argument("--imgsz", type=int, default=640, help="input resolution (default: 640)")
    subparser.add_argument("--output", default=".", help="output directory (default: current)")


def _add_fetch_arguments(subparser):
    subparser.add_argument("target", help="target to fetch the runtime and toolchain for, e.g. cpu")
    subparser.add_argument("--output", default=".", help="output directory (default: current)")


def _add_convert_arguments(subparser):
    subparser.add_argument("model", help="exported model to convert, e.g. yolov8n.onnx")
    subparser.add_argument("--target", required=True, help="target to convert for, e.g. cpu")
    subparser.add_argument("--output", default=".", help="output directory (default: current)")


def _add_detect_arguments(subparser):
    subparser.add_argument("image", help="image to run detection on")
    subparser.add_argument("--target", required=True, help="target to run on, e.g. cpu")
    subparser.add_argument("--model", required=True, help="converted model to load")
    subparser.add_argument(
        "--runtime",
        help="runtime library to load (default: the DEPLOY_ULTRALYTICS_RUNTIME environment variable)",
    )
    subparser.add_argument("--output", default=".", help="output directory (default: current)")
