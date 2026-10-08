"""Chat dispatch for single-image uploads — validation and `create_image`.

Validation happens in the controller before dispatch (Task 10 ruling);
tests hit the real `POST /chat` endpoint with multipart uploads.
"""

import io
import re

import pytest
from PIL import Image

from app.main import app
from app.models import Table, TableColumn

# Byte-exact copy of the brief's upload-validation error.
TOO_BIG = "That file is too big / unsupported — png, jpeg, or webp up to 10 MB."

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@pytest.fixture(autouse=True)
def _no_gemini_key(monkeypatch):
    """Escalation must degrade locally in tests — never hit the real
    Gemini API (mirrors test_chat's fixture)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _page_token(client) -> str:
    """Load GET / and pull the per-page hidden token from the chat form."""
    body = client.get("/").text
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "page must render a hidden token input with a non-empty value"
    return match.group(1)


def _tiny_png() -> bytes:
    """A real 1×1 PNG — content-type is checked, bytes are mocked OCR-side."""
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_oversized_upload_bubbles(client):
    # Review Focus #4: 11 MB png ⇒ HTTP 200 + exact copy, nothing dispatched.
    token = _page_token(client)
    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)

    response = client.post(
        "/chat",
        data={"token": token, "message": "make a wheel"},
        files=[("files", ("big.png", oversized, "image/png"))],
    )

    assert response.status_code == 200
    assert TOO_BIG in response.text
    session = app.state.store.get(token)
    assert session is not None
    assert session.entries == []


def test_gif_rejected(client):
    # Review Focus #4: unsupported type ⇒ same exact copy.
    token = _page_token(client)

    response = client.post(
        "/chat",
        data={"token": token, "message": "make a wheel"},
        files=[("files", ("anim.gif", b"GIF89a-fake-bytes", "image/gif"))],
    )

    assert response.status_code == 200
    assert TOO_BIG in response.text
    session = app.state.store.get(token)
    assert session is not None
    assert session.entries == []


def test_valid_png_builds_wheel_from_image(client, monkeypatch):
    # Mocked OCR (monkeypatch on services.images.read_table), real
    # multipart POST with a real 1×1 png.
    table = Table(
        columns=[
            TableColumn(header="Name", values=["Ana", "Bob"]),
            TableColumn(header="ID", values=["A1", "B2"]),
        ],
        confidence=95.0,
    )
    monkeypatch.setattr("app.services.images.read_table", lambda image_bytes: table)
    token = _page_token(client)

    response = client.post(
        "/chat",
        data={"token": token, "message": "make a wheel with the names"},
        files=[("files", ("roster.png", _tiny_png(), "image/png"))],
    )

    assert response.status_code == 200
    session = app.state.store.get(token)
    assert session is not None
    assert [entry.label for entry in session.entries] == ["Ana", "Bob"]
    assert session.source_table is not None
    assert session.picked == []
    assert "hx-swap-oob" in response.text
