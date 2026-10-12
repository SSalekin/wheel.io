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


def read_table(image_bytes: bytes) -> Table | None:
    """Decode ``image_bytes`` and return the detected table.

    Pipeline: decode (cv2) → grayscale → upscale ×2 if width < 1000 →
    Tesseract word extraction → line grouping → column split → cells.
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
