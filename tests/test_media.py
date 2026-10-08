"""Tests for `GET /media/{token}/{media_id}` — in-memory session images."""

import re

from app.main import app

# Minimal PNG magic-bytes payload; the route serves bytes verbatim.
PNG_BYTES = b"\x89PNG\r\n\x1a\nwheelio-crop"


def _page_token(client) -> str:
    """Load GET / and pull the per-page hidden token from the chat form."""
    body = client.get("/").text
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "page must render a hidden token input with a non-empty value"
    return match.group(1)


def test_media_serves_png(client):
    token = _page_token(client)
    session = app.state.store.get(token)
    assert session is not None
    session.media["crop-1"] = PNG_BYTES

    response = client.get(f"/media/{token}/crop-1")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == PNG_BYTES


def test_media_unknown_404(client):
    token = _page_token(client)
    session = app.state.store.get(token)
    assert session is not None
    session.media["crop-1"] = PNG_BYTES

    # Unknown media id, foreign token, and entirely unknown token all 404.
    assert client.get(f"/media/{token}/missing").status_code == 404
    assert client.get(f"/media/other-token/crop-1").status_code == 404
    assert client.get("/media/nope/nope").status_code == 404
