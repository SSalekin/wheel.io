"""Tests for `app.services.mime.detect_mime` — magic-byte sniffing.

The media route labels stored session images with whatever this returns,
so a JPEG upload must be served as ``image/jpeg``, not ``image/png``.
"""

from app.services.mime import detect_mime


def test_detects_png():
    assert detect_mime(b"\x89PNG\r\n\x1a\nrest") == "image/png"


def test_detects_jpeg():
    assert detect_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"


def test_detects_webp():
    # RIFF container with WEBP at bytes 8-12.
    assert detect_mime(b"RIFF\x10\x00\x00\x00WEBPrest") == "image/webp"


def test_unknown_bytes_default_to_png():
    assert detect_mime(b"mystery-bytes") == "image/png"
    assert detect_mime(b"") == "image/png"
    # A WEBP container without the WEBP kind is not a webp image.
    assert detect_mime(b"RIFF\x10\x00\x00\x00WAVErest") == "image/png"
