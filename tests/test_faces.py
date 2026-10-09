"""Tests for the faces branch — `app.services.faces` + `process_single_image`.

Detection is mocked at its `images.py` import site so the pipeline's
steering/escalation decisions run with no network; two real-cascade
smoke tests at the bottom pin the dependency itself (the cascade asset
must ship with the installed opencv wheel). Cropping runs the real code
against a synthetic image whose pixels encode their own coordinates.
"""

import os

import cv2
import numpy as np
import pytest

from app.models import WheelSession
from app.services import faces, images
from app.services.errors import UserFacingError
from app.services.gemini import GeminiError

# Byte-exact copy of the brief's unreadable-image error.
UNREADABLE = "Couldn't find anything readable in that image."


@pytest.fixture(autouse=True)
def _no_gemini_key(monkeypatch):
    """Escalation must degrade locally in tests — never hit the real
    Gemini API (mirrors test_images's fixture)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def session() -> WheelSession:
    return WheelSession(token="test-token", created_at=0.0)


def _gradient_png(width: int = 100, height: int = 100):
    """Synthetic BGR image where each pixel stores its own (x, y)."""
    y, x = np.mgrid[0:height, 0:width]
    image = np.dstack(
        [x % 256, y % 256, np.full_like(x, 128)]
    ).astype(np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok, "synthetic image must encode"
    return image, buf.tobytes()


def test_crop_faces_margins():
    image, data = _gradient_png()

    # Box (20,20,40,40): ~10% margin per side (frozen interface) = 4px
    # each side ⇒ (16,16)-(64,64). (The brief's ≈ example writes
    # (16,16,68,68); left/top match exactly — symmetric 10% per side is
    # what the Interfaces block pins, so right/bottom are 4px tighter.)
    crops = faces.crop_faces(data, [(20, 20, 40, 40)])

    assert len(crops) == 1
    assert crops[0].startswith(b"\x89PNG"), "crop must be PNG-encoded"
    out = cv2.imdecode(np.frombuffer(crops[0], np.uint8), cv2.IMREAD_COLOR)
    assert out is not None, "crop must decode as PNG"
    assert out.shape[:2] == (48, 48)
    # Corner pixels must come from (16,16) and (63,63) of the source —
    # pins both the margin and the offset, not just the size.
    assert out[0, 0].tolist() == image[16, 16].tolist()
    assert out[47, 47].tolist() == image[63, 63].tolist()

    # Clipped to bounds: a box hanging off the far corner stops at 100×100
    # ⇒ (86,86)-(100,100).
    clipped = faces.crop_faces(data, [(90, 90, 40, 40)])
    assert len(clipped) == 1
    out = cv2.imdecode(np.frombuffer(clipped[0], np.uint8), cv2.IMREAD_COLOR)
    assert out is not None
    assert out.shape[:2] == (14, 14)
    assert out[0, 0].tolist() == image[86, 86].tolist()


def test_detect_boxes_mocked_in_images(monkeypatch, session):
    # Mock detection at its images.py import site — one real-cascade
    # smoke test lives at the bottom of this file. The image is a real
    # synthetic PNG so the real crop_faces runs; read_table is mocked to
    # keep the photo-like input (and tesseract) out of the loop.
    _, image_bytes = _gradient_png(width=200, height=100)
    monkeypatch.setattr(images, "read_table", lambda _bytes: None)
    monkeypatch.setattr(
        images,
        "detect_face_boxes",
        lambda image_bytes: [(10, 10, 50, 50), (80, 10, 50, 50)],
    )
    monkeypatch.setattr(
        "app.services.gemini.ask_vision",
        lambda *args, **kwargs: pytest.fail(
            "ask_vision ran — faces must decide first"
        ),
    )
    monkeypatch.setattr(
        "app.services.gemini.ask_intent",
        lambda *args, **kwargs: pytest.fail(
            "ask_intent ran — steer must decide first"
        ),
    )

    result = images.process_single_image(image_bytes, "wheel the faces", session)

    assert [entry.label for entry in result.entries] == ["Face 1", "Face 2"]
    assert all(entry.image_ref for entry in result.entries)
    assert result.table is None
    assert result.warnings == []
    # Both crops landed in session.media, keyed by the entries' image_ref.
    assert len(session.media) == 2
    assert set(session.media) == {entry.image_ref for entry in result.entries}
    assert all(crop.startswith(b"\x89PNG") for crop in session.media.values())


def test_zero_faces_falls_back_to_vision(monkeypatch, session):
    monkeypatch.setattr(images, "detect_face_boxes", lambda image_bytes: [])
    monkeypatch.setattr(
        "app.services.gemini.ask_vision", lambda image, message: ["Ana"]
    )

    result = images.process_single_image(b"png-bytes", "wheel the faces", session)

    # Vision escalation (same as the text path): plain entry, no crop.
    assert [entry.label for entry in result.entries] == ["Ana"]
    assert result.entries[0].image_ref is None
    assert session.media == {}
    assert result.table is None
    assert result.warnings == []


def test_zero_faces_and_vision_down_raises_unreadable(monkeypatch, session):
    # Review Focus #3: no detections and no vision ⇒ exact user-facing copy.
    def broken_vision(image, message):
        raise GeminiError("vision down")

    monkeypatch.setattr(images, "detect_face_boxes", lambda image_bytes: [])
    monkeypatch.setattr("app.services.gemini.ask_vision", broken_vision)

    with pytest.raises(UserFacingError) as excinfo:
        images.process_single_image(b"png-bytes", "wheel the faces", session)

    assert excinfo.value.user_message == UNREADABLE
    assert str(excinfo.value) == UNREADABLE


def test_detect_face_boxes_wiring(monkeypatch):
    # Review follow-up: the discriminating test for detect_face_boxes —
    # cascade path, grayscale input, int-tuple output. A stub that always
    # returns [], a typo'd cascade filename at the call site, or a skipped
    # gray conversion must fail here (the pipeline tests monkeypatch
    # detection by design, so this is the wiring's only coverage).
    _, image_bytes = _gradient_png(width=320, height=240)
    recorded = {}

    class _SpyCascade:
        def __init__(self, path):
            recorded.setdefault("paths", []).append(path)

        def empty(self):
            return False

        def detectMultiScale(self, image, **kwargs):
            recorded["image"] = image
            return np.array([[10, 10, 50, 50]], dtype=np.int64)

    monkeypatch.setattr(cv2, "CascadeClassifier", _SpyCascade)

    boxes = faces.detect_face_boxes(image_bytes)

    # Built from faces' own constant: a filename typo can't ship green.
    expected_path = os.path.join(cv2.data.haarcascades, faces._CASCADE_FILE)
    assert expected_path in recorded["paths"]
    gray = recorded["image"]
    assert gray.ndim == 2, "detectMultiScale must receive a gray image"
    assert gray.shape == (240, 320), "gray conversion must keep dimensions"
    # np.int64 in ⇒ Python int out: the int() casts are load-bearing.
    assert boxes == [(10, 10, 50, 50)]
    assert all(isinstance(value, int) for box in boxes for value in box)


def test_merge_overlapping_boxes():
    # Overlapping boxes are merged, keeping the larger one
    boxes = [(10, 10, 50, 50), (12, 12, 48, 48), (100, 100, 50, 50)]
    merged = faces._merge_overlapping_boxes(boxes)
    assert len(merged) == 2
    assert (10, 10, 50, 50) in merged
    assert (100, 100, 50, 50) in merged


def test_merge_overlapping_boxes_empty():
    assert faces._merge_overlapping_boxes([]) == []


def test_iou():
    # Identical boxes have IoU = 1.0
    assert faces._iou((10, 10, 50, 50), (10, 10, 50, 50)) == 1.0
    # Non-overlapping boxes have IoU = 0.0
    assert faces._iou((10, 10, 50, 50), (100, 100, 50, 50)) == 0.0
    # Partially overlapping boxes have IoU between 0 and 1
    iou = faces._iou((10, 10, 50, 50), (30, 30, 50, 50))
    assert 0.0 < iou < 1.0


def test_real_cascade_asset_present():
    # Dep ruling: the cascade XML must ship with the installed opencv wheel.
    # Path built from faces' own constant so a typo can't ship green.
    path = os.path.join(cv2.data.haarcascades, faces._CASCADE_FILE)
    assert not cv2.CascadeClassifier(path).empty()


def test_detect_face_boxes_blank_image_returns_empty():
    # Real cascade, deterministic: a uniform image contains no faces.
    blank = np.full((200, 200, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    assert faces.detect_face_boxes(buf.tobytes()) == []


def test_detect_face_boxes_unreadable_bytes_returns_empty():
    assert faces.detect_face_boxes(b"not an image") == []
