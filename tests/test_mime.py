"""Tests for `app.services.mime.detect_mime` — magic-byte sniffing.

The media route labels stored session images with whatever this returns,
so a JPEG upload must be served as ``image/jpeg``, not ``image/png``.
"""

from app.services.mime import detect_mime, is_supported_image


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


def test_is_supported_image_accepts_the_three_formats():
    assert is_supported_image(b"\x89PNG\r\n\x1a\nrest")
    assert is_supported_image(b"\xff\xd8\xff\xe0rest")
    assert is_supported_image(b"RIFF\x10\x00\x00\x00WEBPrest")


def test_is_supported_image_never_guesses():
    # Unlike detect_mime (which defaults to PNG for labeling), unknown
    # bytes are not an image — the upload gate must refuse them.
    assert not is_supported_image(b"<html><body>not an image</body></html>")
    assert not is_supported_image(b"")
    assert not is_supported_image(b"RIFF\x10\x00\x00\x00WAVErest")
