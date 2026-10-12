"""Local face detection for the faces branch (spec §7).

YuNet detection runs entirely offline — the model ships in
``app/models/``. ``detect_face_boxes`` returns face boxes as
``(x, y, w, h)``; ``crop_faces`` turns each box into a standalone PNG
with ~10% margin per side, clipped to the image bounds.

Low-confidence detections (below ``_CONFIDENCE_THRESHOLD``) are
filtered out. If the model fails to load, detection returns ``[]``.
"""

import math
import os
from pathlib import Path

import cv2
import numpy as np

# YuNet model shipped in the app (no runtime download needed).
_MODEL_PATH = str(Path(__file__).parent.parent / "models" / "face_detection_yunet.onnx")

# Minimum confidence for a detection to be kept.
_CONFIDENCE_THRESHOLD = 0.7

# Crop padding: ~10% of each box dimension per side.
_MARGIN_PERCENT = 10

# IoU threshold for merging overlapping detections.
_MERGE_IOU_THRESHOLD = 0.3


def detect_face_boxes(image_bytes: bytes) -> list[tuple[int, int, int, int]]:
    """Detect faces in ``image_bytes`` using YuNet.

    Returns ``(x, y, w, h)`` boxes in pixel coordinates, or ``[]`` when
    the bytes don't decode as an image, the model fails to load, or no
    faces are found above the confidence threshold.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []

    h, w = image.shape[:2]
    detector = _create_detector(w, h)
    if detector is None:
        return []

    try:
        _, detections = detector.detect(image)
    except Exception:
        # YuNet raises on degenerate inputs (an 8×8 PNG once raised
        # OverflowError). "No faces" is a safe answer; a 500 is not.
        return []
    if detections is None:
        return []

    # Filter by confidence and convert to (x, y, w, h) tuples. Rows the
    # model never initialized (huge/NaN coords, zero size) are dropped:
    # a zero-area crop would silently empty the wheel downstream.
    boxes: list[tuple[int, int, int, int]] = []
    for face in detections:
        x, y, width, height = (float(face[0]), float(face[1]), float(face[2]), float(face[3]))
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            continue
        if width <= 0 or height <= 0:
            continue
        confidence = float(face[14])
        if not math.isfinite(confidence) or confidence < _CONFIDENCE_THRESHOLD:
            continue
        boxes.append((int(x), int(y), int(width), int(height)))

    return _merge_overlapping_boxes(boxes)


def _create_detector(w: int, h: int):
    """Create YuNet detector, or return None if model is unavailable."""
    try:
        detector = cv2.FaceDetectorYN.create(_MODEL_PATH, "", (w, h))
        return detector
    except Exception:
        return None


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping boxes, keeping the larger one."""
    if not boxes:
        return []

    # Sort by area (largest first) so we keep the most confident detection
    sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    merged: list[tuple[int, int, int, int]] = []

    for box in sorted_boxes:
        is_duplicate = False
        for kept in merged:
            if _iou(box, kept) > _MERGE_IOU_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(box)

    return merged


def _iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union of two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union if union > 0 else 0.0


def crop_faces(
    image_bytes: bytes, boxes: list[tuple[int, int, int, int]]
) -> list[bytes]:
    """Crop each box out of ``image_bytes`` with ~10% margin per side.

    Crops are clipped to the image bounds and PNG-encoded; a box with no
    visible area after clipping is skipped. Returns ``[]`` when the bytes
    don't decode as an image.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []
    height, width = image.shape[:2]

    crops: list[bytes] = []
    for x, y, w, h in boxes:
        margin_x = max(0, w) * _MARGIN_PERCENT // 100
        margin_y = max(0, h) * _MARGIN_PERCENT // 100
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(width, x + w + margin_x)
        y1 = min(height, y + h + margin_y)
        if x1 <= x0 or y1 <= y0:
            continue
        encoded, buf = cv2.imencode(".png", image[y0:y1, x0:x1])
        if encoded:
            crops.append(buf.tobytes())
    return crops
