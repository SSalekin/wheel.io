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


def test_non_image_bytes_declared_as_png_are_refused(client):
    # Review Focus #4: the bytes are checked, not just the label — an
    # HTML blob declared image/png is refused with the same exact copy.
    token = _page_token(client)
    fake = b"<html><body>not an image</body></html>" * 30000  # ~1 MB, under the cap

    response = client.post(
        "/chat",
        data={"token": token, "message": "wheel"},
        files=[("files", ("fake.png", fake, "image/png"))],
    )

    assert response.status_code == 200
    assert TOO_BIG in response.text


def test_read_validated_caps_an_unbounded_body_with_bounded_reads():
    # A body larger than memory must be refused from chunked reads — the
    # reader never read-all()s and bails just past the cap.
    from app.controllers.chat import _read_validated
    from app.services.errors import UserFacingError

    class _Infinite:
        """An upload whose read() never returns empty."""

        content_type = "image/png"

        def __init__(self):
            self.sizes = []
            self.file = self

        def read(self, size):
            self.sizes.append(size)
            return b"x" * size

    upload = _Infinite()

    with pytest.raises(UserFacingError) as excinfo:
        _read_validated([upload])

    assert excinfo.value.user_message == TOO_BIG
    # Every read is a bounded chunk (never the whole body), and the bail
    # happens within one extra chunk past the cap.
    assert all(size > 0 for size in upload.sizes)
    total_requested = len(upload.sizes) * upload.sizes[0]
    assert MAX_UPLOAD_BYTES < total_requested <= MAX_UPLOAD_BYTES + 1024 * 1024


def test_too_many_files_are_refused(client):
    token = _page_token(client)
    files = [("files", (f"f{index}.png", _tiny_png(), "image/png")) for index in range(21)]

    response = client.post(
        "/chat", data={"token": token, "message": "wheel"}, files=files
    )

    assert response.status_code == 200
    assert TOO_BIG in response.text


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
