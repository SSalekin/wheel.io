"""Local OCR: Tesseract word extraction and table reconstruction.

This is the only module that runs Tesseract. ``read_table`` decodes and
normalizes the image, then pure helpers do the layout work — grouping
word boxes into lines and splitting columns by vertical projection — so
the heuristics stay testable without the tesseract binary.
"""

import cv2
import numpy as np
import pytesseract

from app.models import Table, TableColumn

LANG = "eng"
CONFIG = "--psm 6"
# An empty strip this wide (or wider) in the vertical projection separates
# two columns; narrower strips are just inter-word slack.
MIN_COLUMN_GAP_PX = 20
# Small images get a ×2 boost before OCR for better recognition.
UPSCALE_BELOW_WIDTH = 1000
# Rows darker than this mean form an inverted band (light-on-dark
# headers, dark-mode tables). Full-page binarization drowns such text
# — it read a white-on-navy header row as garbage ('a', 'ee'), which
# then false-matched the column picker.
DARK_BAND_MEAN = 128
# A dark run shorter than this is a divider line, not a text band.
MIN_BAND_HEIGHT_PX = 8
# Dark bands get their own ×3 boost — their text is usually small.
BAND_UPSCALE = 3


def read_table(image_bytes: bytes) -> Table | None:
    """Decode ``image_bytes`` and return the detected table.

    Pipeline: decode (cv2) → grayscale → upscale ×2 if width < 1000 →
    Tesseract word extraction → dark-band pass (inverted headers) →
    line grouping → column split → cells.
    Returns ``None`` when the image yields no readable words.
    """
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 1
    if gray.shape[1] < UPSCALE_BELOW_WIDTH:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        scale = 2
    words = _extract_words(gray)
    # Dark bands (white-on-navy headers) get a second, band-local pass:
    # its words replace whatever full-page OCR hallucinated in the band.
    bands = _dark_bands(gray)
    words = _merge_words(words, _band_words(gray, bands), bands)
    # The gap threshold means REAL pixels: in upscaled coordinates the
    # same real space is twice as wide, so the threshold doubles with it.
    return _build_table(words, min_gap=MIN_COLUMN_GAP_PX * scale)


def _extract_words(image: np.ndarray) -> list[dict]:
    """Run Tesseract and keep only real words: non-empty text, conf >= 0."""
    data = pytesseract.image_to_data(
        image, lang=LANG, config=CONFIG, output_type=pytesseract.Output.DICT
    )
    words = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        conf = int(float(data["conf"][index]))
        if not text or conf < 0:
            continue
        words.append(
            {
                "text": text,
                "conf": conf,
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
            }
        )
    return words


def _dark_bands(gray: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of dark rows at least ``MIN_BAND_HEIGHT_PX`` tall.

    These are inverted regions — light-on-dark header bands, dark-mode
    tables — whose text full-page binarization cannot read.
    """
    dark_rows = gray.mean(axis=1) < DARK_BAND_MEAN
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y, is_dark in enumerate(dark_rows):
        if is_dark and start is None:
            start = y
        elif not is_dark and start is not None:
            if y - start >= MIN_BAND_HEIGHT_PX:
                bands.append((start, y))
            start = None
    if start is not None and len(dark_rows) - start >= MIN_BAND_HEIGHT_PX:
        bands.append((start, len(dark_rows)))
    return bands


def _band_words(gray: np.ndarray, bands: list[tuple[int, int]]) -> list[dict]:
    """Words OCR'd band-locally: Otsu on the band strip alone.

    A dark strip's histogram is dominated by its background, so a
    band-local threshold separates the light text that the full-page
    pass drowns. Coordinates are scaled back into ``gray``'s space.
    """
    words: list[dict] = []
    for start, end in bands:
        strip = gray[start:end]
        _, binary = cv2.threshold(
            strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        big = cv2.resize(
            binary, None, fx=BAND_UPSCALE, fy=BAND_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        for word in _extract_words(big):
            word["left"] //= BAND_UPSCALE
            word["top"] = word["top"] // BAND_UPSCALE + start
            word["width"] //= BAND_UPSCALE
            word["height"] //= BAND_UPSCALE
            words.append(word)
    return words


def _merge_words(
    page_words: list[dict],
    band_words: list[dict],
    bands: list[tuple[int, int]],
) -> list[dict]:
    """Band OCR is authoritative inside dark bands.

    Full-page binarization cannot read light-on-dark text, so whatever
    it produced over a band's rows — conf-15 'a'/'ee' fragments that
    would false-match the column picker — is dropped; words elsewhere
    (body rows) stay untouched.
    """
    merged = []
    for word in page_words:
        bottom = word["top"] + word["height"]
        over_band = any(
            word["top"] < end and bottom > start for start, end in bands
        )
        if not over_band:
            merged.append(word)
    return merged + band_words


def _group_lines(words: list[dict]) -> list[list[dict]]:
    """Group word boxes into rows by vertical overlap; left-to-right inside."""
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: w["top"]):
        bottom = word["top"] + word["height"]
        if lines:
            line = lines[-1]
            line_top = min(w["top"] for w in line)
            line_bottom = max(w["top"] + w["height"] for w in line)
            if word["top"] < line_bottom and bottom > line_top:
                line.append(word)
                continue
        lines.append([word])
    for line in lines:
        line.sort(key=lambda w: w["left"])
    return lines


def _column_segments(
    words: list[dict], min_gap: float = MIN_COLUMN_GAP_PX
) -> list[tuple[int, int]]:
    """Split word boxes into columns by vertical projection.

    Occupancy is computed per x-column across all rows; text runs are
    merged while the empty strip between them stays below ``min_gap``
    (``MIN_COLUMN_GAP_PX`` of real image space — the caller scales it
    when the image was upscaled). Returns half-open ``[start, end)``
    x-ranges; no words ⇒ no segments.
    """
    if not words:
        return []
    right = max(w["left"] + w["width"] for w in words)
    occupancy = np.zeros(right + 1, dtype=bool)
    for word in words:
        occupancy[word["left"] : word["left"] + word["width"]] = True

    segments: list[tuple[int, int]] = []
    x = 0
    while x <= right:
        if not occupancy[x]:
            x += 1
            continue
        start = x
        while x <= right and occupancy[x]:
            x += 1
        # x is the first empty column after this text run (or past the end).
        if segments and start - segments[-1][1] < min_gap:
            segments[-1] = (segments[-1][0], x)  # strip too narrow to split
        else:
            segments.append((start, x))
    return segments


def _build_table(
    words: list[dict], min_gap: float = MIN_COLUMN_GAP_PX
) -> Table | None:
    """Assemble word boxes into a ``Table``.

    Multi-column tables take the first line as the header; a
    single-column table is a headerless list — a photo of plain names —
    so every line, including the first, is a value.
    """
    if not words:
        return None
    lines = _group_lines(words)
    segments = _column_segments(words, min_gap)

    rows: list[list[str]] = []
    for line in lines:
        cells = [""] * len(segments)
        for word in line:
            center = word["left"] + word["width"] / 2
            for index, (start, end) in enumerate(segments):
                if start <= center < end:
                    text = cells[index]
                    cells[index] = f"{text} {word['text']}" if text else word["text"]
                    break
        rows.append(cells)

    confidence = sum(word["conf"] for word in words) / len(words)
    if len(segments) == 1:
        # Headerless single-column list — a photo of plain names: every
        # line is a value. Consuming line 1 as a header would silently
        # drop the first name (audit #3).
        return Table(
            columns=[TableColumn(header="", values=[row[0] for row in rows])],
            confidence=confidence,
        )
    headers = rows[0]
    columns = [
        TableColumn(header=header, values=[row[index] for row in rows[1:]])
        for index, header in enumerate(headers)
    ]
    return Table(columns=columns, confidence=confidence)
