# Fixtures

## test-image.jpg

"Calgary Car Company Bus" by Bernard Spragg. NZ, 2012, published under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication), so it can
be redistributed under this repository's Apache-2.0 license without attribution.

- Source page: <https://commons.wikimedia.org/wiki/File:Calgary_Car_Company_Bus_(10471166024).jpg>
- Original: <https://www.flickr.com/photos/volvob12b/10471166024/>
- Modification: the 960 px wide rendition served by Wikimedia Commons, unedited otherwise.

It shows people and a truck, which are COCO classes.

## test-image.yolov8n.json

The reference detections for `yolov8n.pt` on `test-image.jpg`: what Ultralytics' own `predict`
returns, independent of OAAX. Every target's `detect` output is compared against it.

```json
{
  "model": "yolov8n.pt",
  "image": "test-image.jpg",
  "detections": [{ "box": [x1, y1, x2, y2], "class": 0, "score": 0.855 }]
}
```

- `box`: corners in pixels of the original image, rounded to 0.1.
- `class`: COCO class index, the same index as in the model manifest's class names.
- `score`: confidence, rounded to 3 decimals.
- Detections are ordered by descending score. Compare by class and box IoU, not by exact values.

Regenerate it after changing the model, the image or the `ultralytics` version:

```sh
python -m deploy_ultralytics.reference yolov8n.pt fixtures/test-image.jpg
```

The output lands next to the image as `<image stem>.<model stem>.json`; pass `-o` to choose
another path. `tests/test_reference_detections.py` fails when the committed file differs from a
fresh regeneration.
