"""Regenerate the reference detections: what Ultralytics itself predicts for a model and image.

python -m deploy_ultralytics.reference yolov8n.pt fixtures/test-image.jpg
"""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    image = Path(args.image)
    output = Path(args.output) if args.output else image.with_suffix(f".{Path(args.model).stem}.json")
    output.write_text(json.dumps(reference_detections(args.model, image), indent=2) + "\n")
    return 0


def reference_detections(model_name: str, image: str | Path) -> dict:
    """Run the source model through Ultralytics' own `predict` on the image.

    Boxes are rounded to 0.1 px and scores to 3 decimals so that the fixture does not churn on
    floating-point noise across platforms.
    """
    from ultralytics import YOLO

    result = YOLO(model_name).predict(str(image), verbose=False)[0]
    detections = [
        {
            "box": [round(v, 1) for v in box],
            "class": int(cls),
            "score": round(score, 3),
        }
        for box, cls, score in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True
        )
    ]
    return {"model": model_name, "image": Path(image).name, "detections": detections}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deploy_ultralytics.reference",
        description="Write the reference detections Ultralytics produces for a model and image.",
    )
    parser.add_argument("model", help="Ultralytics model name, e.g. yolov8n.pt")
    parser.add_argument("image", help="test image to run the model on")
    parser.add_argument(
        "-o",
        "--output",
        help="where to write the JSON (default: <image stem>.<model stem>.json next to the image)",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
