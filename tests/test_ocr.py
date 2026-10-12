"""Tests for the Tesseract OCR pipeline — `app.services.ocr.read_table`.

Unit tests mock `pytesseract.image_to_data` with fixed word boxes so they
never need the tesseract binary. Integration tests render a real PNG with
PIL and skip when no tesseract binary is on PATH.
"""

import io
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.services.ocr import read_table

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="tesseract binary not found on PATH",
)


def _fake_data(words: list[tuple]) -> dict:
    """Build the DICT pytesseract.image_to_data returns.

    Each word is a (text, conf, left, top, width, height) tuple.
    """
    return {
        "text": [word[0] for word in words],
        "conf": [word[1] for word in words],
        "left": [word[2] for word in words],
        "top": [word[3] for word in words],
        "width": [word[4] for word in words],
        "height": [word[5] for word in words],
    }


def _patch_image_to_data(monkeypatch, data: dict, capture: dict | None = None):
    import pytesseract

    def fake(image, **kwargs):
        if capture is not None:
            capture.update(kwargs)
        return data

    monkeypatch.setattr(pytesseract, "image_to_data", fake)


def _blank_png(width: int = 120, height: int = 80) -> bytes:
    """Any decodable image — unit tests never run OCR on it."""
    buffer = io.BytesIO()
    Image.new("L", (width, height), 255).save(buffer, format="PNG")
    return buffer.getvalue()


def test_single_column_table_keeps_every_line_as_data(monkeypatch):
    # Audit #3: a photo of a plain name list has no header row — the first
    # line is a name like any other, not a column header to be consumed.
    _patch_image_to_data(
        monkeypatch,
        _fake_data(
            [
                ("Ana", 95, 10, 10, 80, 30),
                ("Bob", 95, 10, 60, 80, 30),
                ("Carol", 95, 10, 110, 80, 30),
            ]
        ),
    )

    table = read_table(_blank_png())

    assert table is not None
    assert len(table.columns) == 1
    assert table.columns[0].header == ""
    assert table.columns[0].values == ["Ana", "Bob", "Carol"]


def test_words_group_into_lines_and_columns(monkeypatch):
    # Two column blocks with a 110 px empty strip between them; three rows
    # whose vertical ranges (10-40, 60-90, 110-140) never overlap.
    data = _fake_data(
        [
            ("Name", 95, 10, 10, 80, 30),
            ("ID", 95, 200, 10, 40, 30),
            ("Ana", 90, 10, 60, 70, 30),
            ("A1", 90, 200, 60, 50, 30),
            ("Bob", 90, 10, 110, 70, 30),
            ("B2", 90, 200, 110, 50, 30),
        ]
    )
    capture: dict = {}
    _patch_image_to_data(monkeypatch, data, capture)

    table = read_table(_blank_png())

    assert table is not None
    assert table.headers == ["Name", "ID"]
    assert table.get_column("Name").values == ["Ana", "Bob"]
    assert table.get_column("ID").values == ["A1", "B2"]
    # Interfaces contract: eng + uniform-block segmentation.
    assert capture["lang"] == "eng"
    assert capture["config"] == "--psm 6"


def test_empty_result_returns_none(monkeypatch):
    # Blank input comes back as empty-text, conf -1 level rows — no words kept.
    data = _fake_data([("", -1, 0, 0, 0, 0), ("", -1, 0, 0, 0, 0)])
    _patch_image_to_data(monkeypatch, data)

    assert read_table(_blank_png()) is None


def test_confidence_is_mean(monkeypatch):
    data = _fake_data(
        [
            ("Name", 80, 10, 10, 80, 30),
            ("ID", 60, 200, 10, 40, 30),
        ]
    )
    _patch_image_to_data(monkeypatch, data)

    table = read_table(_blank_png())

    assert table is not None
    assert table.confidence == 70.0


def _render_roster() -> bytes:
    """White PNG: header Name/ID, rows Ana/A1 and Bob/B2, 60 px column gap."""
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    rows = [("Name", "ID"), ("Ana", "A1"), ("Bob", "B2")]
    probe = ImageDraw.Draw(Image.new("L", (8, 8), 255))
    col2_x = int(40 + max(probe.textlength(left, font=font) for left, _ in rows) + 60)
    image = Image.new("RGB", (col2_x + 200, 320), "white")
    draw = ImageDraw.Draw(image)
    for index, (left, right) in enumerate(rows):
        y = 30 + index * 70
        draw.text((40, y), left, fill="black", font=font)
        draw.text((col2_x, y), right, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@needs_tesseract
def test_synthetic_roster_roundtrip():
    table = read_table(_render_roster())

    assert table is not None
    assert table.headers == ["Name", "ID"]
    assert table.get_column("Name").values == ["Ana", "Bob"]
    assert table.confidence > 60


@needs_tesseract
def test_blank_image_returns_none():
    buffer = io.BytesIO()
    Image.new("RGB", (400, 200), "white").save(buffer, format="PNG")

    assert read_table(buffer.getvalue()) is None


def test_column_segments_empty_words_returns_empty():
    # An image with no readable words must not crash the projection
    # (max() over an empty sequence).
    from app.services.ocr import _column_segments

    assert _column_segments([]) == []


def test_column_gap_threshold_scales_with_upscale(monkeypatch):
    # The same 24 px gap in tesseract coords is 12 px of real space in a
    # small image upscaled ×2 — below MIN_COLUMN_GAP_PX (20) of real
    # space, so it is letter-spacing, not a column break. In a
    # native-width image the same gap is a real split.
    _patch_image_to_data(
        monkeypatch,
        _fake_data(
            [
                ("Ana", 95, 10, 10, 60, 30),
                ("Bob", 95, 94, 10, 60, 30),  # gap x=70..93 (24 px)
            ]
        ),
    )

    native = read_table(_blank_png(width=1200, height=200))
    upscaled = read_table(_blank_png(width=400, height=200))

    assert native is not None and len(native.columns) == 2
    assert upscaled is not None and len(upscaled.columns) == 1


def _render_inverted_header() -> bytes:
    """PNG with a white-on-navy header band, like real dashboard exports.

    Reported bug image: the header row sat on a dark band and full-page
    binarization rendered it as garbage ('a', 'ee'), starving every
    downstream column match.
    """
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    image = Image.new("RGB", (1100, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1100, 70], fill=(40, 40, 120))
    draw.text((40, 15), "Name", fill="white", font=font)
    draw.text((400, 15), "ID", fill="white", font=font)
    for index, (name, ident) in enumerate([("Ana", "A1"), ("Bob", "B2")]):
        y = 100 + index * 70
        draw.text((40, y), name, fill="black", font=font)
        draw.text((400, y), ident, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@needs_tesseract
def test_inverted_header_row_roundtrip():
    # White-on-navy headers must read as headers, not as garbage
    # fragments that false-match the column picker ('a' in any message).
    table = read_table(_render_inverted_header())

    assert table is not None
    assert table.headers == ["Name", "ID"]
    assert table.get_column("Name").values == ["Ana", "Bob"]


def test_dark_bands_finds_contiguous_dark_rows():
    # A dark header band is a run of dark rows; a thin dark divider
    # line is not worth OCRing.
    import numpy as np

    from app.services.ocr import _dark_bands

    gray = np.full((100, 200), 255, dtype=np.uint8)
    gray[10:40] = 40
    gray[60:63] = 30

    assert _dark_bands(gray) == [(10, 40)]


def test_band_words_replace_garbage_in_dark_bands():
    # Band OCR is authoritative over a band's ROWS: fragments full-page
    # OCR produced there (conf-15 'a'/'ee' noise) must not survive to
    # false-match the picker — even when they sit beside a real band
    # word; body words below the band stay untouched.
    from app.services.ocr import _merge_words

    page = [
        {"text": "ee", "conf": 32, "left": 460, "top": 10, "width": 30, "height": 25},
        {"text": "NhungTTH37", "conf": 65, "left": 520, "top": 120, "width": 120, "height": 25},
    ]
    band = [{"text": "Lecturer", "conf": 96, "left": 500, "top": 8, "width": 140, "height": 30}]

    merged = _merge_words(page, band, bands=[(5, 45)])

    assert sorted(word["text"] for word in merged) == ["Lecturer", "NhungTTH37"]
