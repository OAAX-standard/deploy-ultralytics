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
    print(f"deploy-ultralytics {args.command}: not implemented yet", file=sys.stderr)
    return 1


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="deploy-ultralytics",
        description="Run Ultralytics YOLO models on several accelerators through OAAX.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{export,fetch,convert,detect}")
    for name, help_text in _SUBCOMMANDS.items():
        subparsers.add_parser(name, help=help_text, description=help_text)
    return parser
