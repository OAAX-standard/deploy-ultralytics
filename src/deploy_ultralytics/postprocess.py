import numpy as np

CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300


def decode(output, letterbox_, image_shape):
    """Turn a `(1, 4 + num_classes, N)` output into detections in original-image pixels.

    Returns a list of `{"box": [x1, y1, x2, y2], "class": int, "score": float}`, best score first.
    """
    predictions = output[0].T
    class_scores = predictions[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)
    keep = scores >= CONFIDENCE_THRESHOLD
    boxes, classes, scores = predictions[keep, :4], classes[keep], scores[keep]

    corners = np.empty_like(boxes)
    corners[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    corners[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2

    kept = non_max_suppression(corners, scores, classes, NMS_IOU_THRESHOLD)[:MAX_DETECTIONS]
    original = letterbox_.to_original(corners[kept], image_shape)
    return [
        {
            "box": [round(float(v), 1) for v in box],
            "class": int(cls),
            "score": round(float(score), 3),
        }
        for box, cls, score in zip(original, classes[kept], scores[kept], strict=True)
    ]


def non_max_suppression(boxes, scores, classes, iou_threshold):
    """Indices of the boxes to keep, best score first; boxes only suppress the same class."""
    order = scores.argsort()[::-1]
    kept = []
    while order.size:
        best, order = order[0], order[1:]
        kept.append(best)
        same_class = classes[order] == classes[best]
        overlap = _iou(boxes[best], boxes[order])
        order = order[~(same_class & (overlap > iou_threshold))]
    return np.array(kept, dtype=int)


def _iou(box, others):
    top_left = np.maximum(box[:2], others[:, :2])
    bottom_right = np.minimum(box[2:], others[:, 2:])
    intersection = np.prod((bottom_right - top_left).clip(0), axis=1)
    area = np.prod(box[2:] - box[:2])
    other_areas = np.prod(others[:, 2:] - others[:, :2], axis=1)
    return intersection / (area + other_areas - intersection)
