"""Tests for the single-image pipeline — `app.services.images.process_single_image`.

OCR (`read_table`) and the Gemini client are mocked at their use sites so
the tests exercise the pipeline's steering and escalation decisions with
no network, no tesseract binary, and byte-exact error copy.
"""

import pytest

from app.models import Table, TableColumn, WheelSession
from app.services import images
from app.services.errors import UserFacingError
from app.services.gemini import GeminiError

# Byte-exact copy of the brief's unreadable-image error.
UNREADABLE = "Couldn't find anything readable in that image."


@pytest.fixture(autouse=True)
def _no_gemini_key(monkeypatch):
    """Escalation must degrade locally in tests — never hit the real
    Gemini API (mirrors test_chat's fixture)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def session() -> WheelSession:
    return WheelSession(token="test-token", created_at=0.0)


def _table(confidence: float = 95.0) -> Table:
    return Table(
        columns=[
            TableColumn(header="Name", values=["Ana", "Bob"]),
            TableColumn(header="ID", values=["A1", "B2"]),
        ],
        confidence=confidence,
    )


def test_text_path_builds_entries(monkeypatch, session):
    # Mock the OCR step (ocr.read_table) at its images.py import site.
    monkeypatch.setattr(images, "read_table", lambda image_bytes: _table())
    # A clear text steer must decide without any tie-break: if steering
    # broke and the message fell through to ask_intent, this stub fails
    # the test instead of letting the GeminiError fallback pass it.
    monkeypatch.setattr(
        "app.services.gemini.ask_intent",
        lambda *args, **kwargs: pytest.fail("ask_intent ran — steer must decide first"),
    )

    result = images.process_single_image(b"png-bytes", "wheel the names", session)

    # Real picker: "names" steers the name-header group, so the Name
    # column wins with source="header" and no Gemini fallback warning.
    assert [entry.label for entry in result.entries] == ["Ana", "Bob"]
    assert result.table is not None
    assert result.warnings == []


def test_low_confidence_escalates_to_vision(monkeypatch, session):
    monkeypatch.setattr(images, "read_table", lambda image_bytes: _table(confidence=40))
    monkeypatch.setattr(
        "app.services.gemini.ask_vision", lambda image, message: ["Zoe"]
    )

    result = images.process_single_image(b"png-bytes", "wheel the names", session)

    # Reading unreliable (confidence 40 < 60) ⇒ vision labels replace the
    # table entirely: source table is not carried on a vision escalation.
    assert [entry.label for entry in result.entries] == ["Zoe"]
    assert result.table is None
    assert result.warnings == []


def test_vision_down_with_unusable_table_raises_unreadable(monkeypatch, session):
    # Review Focus #3: no table and no vision ⇒ exact user-facing copy.
    def broken_vision(image, message):
        raise GeminiError("vision down")

    monkeypatch.setattr(images, "read_table", lambda image_bytes: None)
    monkeypatch.setattr("app.services.gemini.ask_vision", broken_vision)

    with pytest.raises(UserFacingError) as excinfo:
        images.process_single_image(b"png-bytes", "wheel the names", session)

    assert excinfo.value.user_message == UNREADABLE
    assert str(excinfo.value) == UNREADABLE


def test_face_steered_raises_until_task_11(monkeypatch, session):
    # Face-only steer takes the faces branch; Task 10 ships the
    # placeholder raise that Task 11 replaces with real detection.
    # A real table is present on purpose: with steering broken the
    # ambiguous-branch GeminiError fallback sees the table and takes the
    # *text* path — only genuine FACE_STEER matching can raise here.
    monkeypatch.setattr(images, "read_table", lambda image_bytes: _table())

    with pytest.raises(UserFacingError) as excinfo:
        images.process_single_image(b"png-bytes", "wheel the faces", session)

    assert excinfo.value.user_message == UNREADABLE
