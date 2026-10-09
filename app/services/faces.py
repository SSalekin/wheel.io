"""Local face detection for the faces branch (spec §7).

Haar cascade detection runs entirely offline — no AI involved.
``detect_face_boxes`` returns frontal-face boxes as ``(x, y, w, h)``;
``crop_faces`` turns each box into a standalone PNG with ~10% margin per
side, clipped to the image bounds.

Detection tries multiple strategies (strict, relaxed, profile) and merges
results to handle group photos where a single strict pass misses faces.
"""

import os

import cv2
import numpy as np

# Cascade asset shipped inside the opencv wheel (see cv2.data.haarcascades).
_CASCADE_FILE = "haarcascade_frontalface_default.xml"
_PROFILE_CASCADE_FILE = "haarcascade_profileface.xml"

# Detection strategies: (cascade_file, scale_factor, min_neighbors, min_size)
# Tried in order; results are merged with overlap detection.
_DETECTION_STRATEGIES = [
    (_CASCADE_FILE, 1.1, 5, (30, 30)),  # strict frontal
    (_CASCADE_FILE, 1.1, 3, (20, 20)),  # relaxed frontal
    (_PROFILE_CASCADE_FILE, 1.1, 3, (20, 20)),  # relaxed profile
]

# Crop padding: ~10% of each box dimension per side.
_MARGIN_PERCENT = 10

# IoU threshold for merging overlapping detections.
_MERGE_IOU_THRESHOLD = 0.3


def detect_face_boxes(image_bytes: bytes) -> list[tuple[int, int, int, int]]:
    """Detect frontal faces in ``image_bytes``.

    Returns ``(x, y, w, h)`` boxes in pixel coordinates, or ``[]`` when
    the bytes don't decode as an image or nothing is found.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    all_boxes: list[tuple[int, int, int, int]] = []
    for cascade_file, scale_factor, min_neighbors, min_size in _DETECTION_STRATEGIES:
        cascade_path = os.path.join(cv2.data.haarcascades, cascade_file)
        if not os.path.exists(cascade_path):
            continue
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            continue
        boxes = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        all_boxes.extend([(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes])

    return _merge_overlapping_boxes(all_boxes)


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
