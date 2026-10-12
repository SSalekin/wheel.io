"""Tests for `GET /media/{token}/{media_id}` — in-memory session images."""

import re

from app.main import app

# Minimal magic-bytes payloads; the route serves bytes verbatim.
PNG_BYTES = b"\x89PNG\r\n\x1a\nwheelio-crop"
JPEG_BYTES = b"\xff\xd8\xff\xe0wheelio-crop"
WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBPVP8 wheelio-crop"


def _page_token(client) -> str:
    """Load GET / and pull the per-page hidden token from the chat form."""
    body = client.get("/").text
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "page must render a hidden token input with a non-empty value"
    return match.group(1)


def test_media_sets_nosniff(client):
    # Stored bytes are sniffed at serve time; nosniff stops browsers from
    # second-guessing the label into a script context.
    token = _page_token(client)
    session = app.state.store.get(token)
    assert session is not None
    session.media["crop-1"] = PNG_BYTES

    response = client.get(f"/media/{token}/crop-1")

    assert response.headers["x-content-type-options"] == "nosniff"


def test_media_serves_png(client):
    token = _page_token(client)
    session = app.state.store.get(token)
    assert session is not None
    session.media["crop-1"] = PNG_BYTES

    response = client.get(f"/media/{token}/crop-1")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == PNG_BYTES


def test_media_serves_upload_bytes_with_their_own_mime(client):
    # Regression: multi-image uploads kept JPEG/WEBP bytes but were all
    # served as image/png. The bytes are unchanged; only the label is real.
    token = _page_token(client)
    session = app.state.store.get(token)
    assert session is not None
    session.media["photo"] = JPEG_BYTES
    session.media["avatar"] = WEBP_BYTES

    jpeg = client.get(f"/media/{token}/photo")
    webp = client.get(f"/media/{token}/avatar")

    assert jpeg.headers["content-type"] == "image/jpeg"
    assert jpeg.content == JPEG_BYTES
    assert webp.headers["content-type"] == "image/webp"
    assert webp.content == WEBP_BYTES


def test_media_unknown_bytes_still_serve_as_png(client):
    token = _page_token(client)
    session = app.state.store.get(token)
    assert session is not None
    session.media["crop-1"] = b"mystery-bytes"

    response = client.get(f"/media/{token}/crop-1")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"mystery-bytes"


def test_media_unknown_404(client):
    token = _page_token(client)
    session = app.state.store.get(token)
    assert session is not None
    session.media["crop-1"] = PNG_BYTES

    # Unknown media id, foreign token, and entirely unknown token all 404.
    assert client.get(f"/media/{token}/missing").status_code == 404
    assert client.get(f"/media/other-token/crop-1").status_code == 404
    assert client.get("/media/nope/nope").status_code == 404
