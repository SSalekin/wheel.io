"""Local face detection for the faces branch (spec §7).

Haar cascade detection runs entirely offline — no AI involved.
``detect_face_boxes`` returns frontal-face boxes as ``(x, y, w, h)``;
``crop_faces`` turns each box into a standalone PNG with ~10% margin per
side, clipped to the image bounds.
"""

import os

import cv2
import numpy as np

# Cascade asset shipped inside the opencv wheel (see cv2.data.haarcascades).
_CASCADE_FILE = "haarcascade_frontalface_default.xml"

# Standard detectMultiScale params: gentle scale steps, the classic
# neighborhood-vote floor, and a small floor against noise specks.
_SCALE_FACTOR = 1.1
_MIN_NEIGHBORS = 5
_MIN_SIZE = (30, 30)

# Crop padding: ~10% of each box dimension per side.
_MARGIN_PERCENT = 10


def detect_face_boxes(image_bytes: bytes) -> list[tuple[int, int, int, int]]:
    """Detect frontal faces in ``image_bytes``.

    Returns ``(x, y, w, h)`` boxes in pixel coordinates, or ``[]`` when
    the bytes don't decode as an image or nothing is found.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []
    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, _CASCADE_FILE)
    )
    if cascade.empty():  # dep guard: asset must ship with the opencv wheel
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=_SCALE_FACTOR,
        minNeighbors=_MIN_NEIGHBORS,
        minSize=_MIN_SIZE,
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]


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
