"""Tests for the multi-image flow — `app.services.images.process_multiple_images`.

Filenames are the labels (prettified); no detection, no OCR, no Gemini.
The dispatch test hits the real `POST /chat` endpoint with multipart
uploads (validation happens in the controller before dispatch, Task 10).
"""

import io
import re

import pytest
import respx
from PIL import Image

from app.main import app
from app.models import WheelSession
from app.services import images

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-2.5-flash:generateContent"
)


@pytest.fixture(autouse=True)
def _no_gemini_key(monkeypatch):
    """Escalation must degrade locally in tests — never hit the real
    Gemini API (mirrors test_chat's fixture)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def session() -> WheelSession:
    return WheelSession(token="test-token", created_at=0.0)


def _page_token(client) -> str:
    """Load GET / and pull the per-page hidden token from the chat form."""
    body = client.get("/").text
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "page must render a hidden token input with a non-empty value"
    return match.group(1)


def _tiny_png() -> bytes:
    """A real 1×1 PNG — bytes are never parsed by this flow (no OCR)."""
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_each_image_becomes_entry(session):
    named = [
        ("student_a.png", b"first-bytes"),
        ("team photo.png", b"second-bytes"),
        ("Chi.PNG", b"third-bytes"),
    ]

    result = images.process_multiple_images(named, session)

    # Byte-pinned prettify rules: filename minus extension, `[-_]+` → " ",
    # strip, title-case ("Chi.PNG" → "Chi" — already title-cased).
    assert [entry.label for entry in result.entries] == [
        "Student A",
        "Team Photo",
        "Chi",
    ]
    # One media id per file, each entry pointing at its own stored bytes.
    assert len(session.media) == 3
    assert set(session.media) == {entry.image_ref for entry in result.entries}
    assert [session.media[entry.image_ref] for entry in result.entries] == [
        b"first-bytes",
        b"second-bytes",
        b"third-bytes",
    ]
    # Interfaces: no table, no warnings — filenames decide, nothing else.
    assert result.table is None
    assert result.warnings == []


@respx.mock
def test_multi_image_flow_needs_no_gemini(monkeypatch, session):
    # With a key present, any Gemini attempt would reach HTTP and be
    # recorded by respx — assert_no_call fails if the flow ever grows one.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    respx.post(API_URL)

    result = images.process_multiple_images(
        [("ana.png", b"ana-bytes"), ("bob.png", b"bob-bytes")], session
    )

    assert [entry.label for entry in result.entries] == ["Ana", "Bob"]
    respx.calls.assert_not_called()


def test_chat_dispatch_builds_wheel(client):
    token = _page_token(client)

    response = client.post(
        "/chat",
        data={"token": token, "message": "avatars"},
        files=[
            ("files", ("team photo.png", _tiny_png(), "image/png")),
            ("files", ("Chi.PNG", _tiny_png(), "image/png")),
        ],
    )

    assert response.status_code == 200
    session = app.state.store.get(token)
    assert session is not None
    assert len(session.entries) == 2
    assert [entry.label for entry in session.entries] == ["Team Photo", "Chi"]
    # Entries keep their media refs so the wheel can draw thumbnails.
    assert all(entry.image_ref in session.media for entry in session.entries)
    # The regions that change (wheel pane, sidebar, go form) ship OOB.
    assert "hx-swap-oob" in response.text
