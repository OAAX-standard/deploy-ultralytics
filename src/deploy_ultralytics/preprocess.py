from collections.abc import Sequence

import cv2
import numpy as np

from deploy_ultralytics.target import TargetConfig

PAD_VALUE = 114


class Letterbox:
    """How an image was scaled and padded to fit the model input, so boxes can be mapped back."""

    def __init__(self, scale: float, pad_x: int, pad_y: int) -> None:
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y

    def to_original(self, boxes: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
        """Map xyxy boxes from model-input pixels to the original image, clipped to it."""
        height, width = image_shape[:2]
        mapped = (boxes - [self.pad_x, self.pad_y, self.pad_x, self.pad_y]) / self.scale
        mapped[:, [0, 2]] = mapped[:, [0, 2]].clip(0, width)
        mapped[:, [1, 3]] = mapped[:, [1, 3]].clip(0, height)
        return mapped


def letterbox(image: np.ndarray, size: Sequence[int]) -> tuple[np.ndarray, Letterbox]:
    """Scale a BGR image to fit `size` (height, width) keeping its aspect ratio, pad the rest."""
    height, width = image.shape[:2]
    scale = min(size[0] / height, size[1] / width)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # An odd amount of padding leaves the extra pixel on the right or bottom.
    left, top = (size[1] - new_width) // 2, (size[0] - new_height) // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        size[0] - new_height - top,
        left,
        size[1] - new_width - left,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE,) * 3,
    )
    return padded, Letterbox(scale, left, top)


def to_input_tensor(image: np.ndarray, target: TargetConfig) -> np.ndarray:
    """Turn a letterboxed BGR image into the tensor the target's runtime expects."""
    rgb = image[:, :, ::-1].astype(np.float32)
    normalized = (rgb - np.array(target.mean, dtype=np.float32)) / np.array(target.std, dtype=np.float32)
    if target.layout == "NCHW":
        normalized = normalized.transpose(2, 0, 1)
    return np.ascontiguousarray(normalized[np.newaxis].astype(target.element_type))
