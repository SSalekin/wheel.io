"""Image pipelines: single-image read/steer, multi-image filenames.

Spec §7: the table reader runs first, then the message steers text vs
faces (``TEXT_STEER``/``FACE_STEER``) with Gemini breaking ambiguous
ties from detector summaries — it never sees the image. The text path
escalates to Gemini vision whenever the reading itself is unreliable;
a reliable table goes through the column picker instead.

Multiple uploads take the filename-only path
(``process_multiple_images``): no reading, no detection, no Gemini —
each file becomes one entry labeled by its own name.
"""

import re
from pathlib import PurePath
from uuid import uuid4

from pydantic import BaseModel

from app.models import Entry, Intent, Table, WheelSession
from app.services import gemini
from app.services.column_picker import pick_column
from app.services.errors import UserFacingError
from app.services.faces import crop_faces, detect_face_boxes
from app.services.gemini import GeminiError
from app.services.ocr import read_table
from app.services.rules import FACE_STEER, TEXT_STEER

# Spec §7 step 5: escalate when mean word confidence drops below this.
MIN_CONFIDENCE = 60

# Byte-exact user-facing copy (never vary).
_UNREADABLE = "Couldn't find anything readable in that image."
_AI_UNREACHABLE_WARNING = "⚠️ AI unreachable — best-guess parsing."


class ImageResult(BaseModel):
    """What one processed upload contributes to the wheel session."""

    entries: list[Entry]
    table: Table | None
    warnings: list[str]


def process_single_image(
    image_bytes: bytes, message: str, session: WheelSession
) -> ImageResult:
    """Turn one upload into wheel entries (spec §7 single-image pipeline).

    ``session`` travels along for the faces branch: detected face crops
    are stored in ``session.media`` and referenced by ``image_ref``.
    """
    table = read_table(image_bytes)

    tokens = set(message.lower().split())
    text_steer = bool(tokens & TEXT_STEER)
    face_steer = bool(tokens & FACE_STEER)

    if face_steer and not text_steer:
        return _faces_branch(image_bytes, message, session)
    if text_steer and not face_steer:
        return _text_path(image_bytes, message, table)

    # Ambiguous steer (both or neither): run detection first.
    # If faces are detected, go directly to the faces branch — no need
    # to ask Gemini. Only escalate to Gemini when no faces are found.
    face_boxes = detect_face_boxes(image_bytes)
    if face_boxes:
        return _faces_branch(image_bytes, message, session, face_boxes)

    try:
        intent = gemini.ask_intent(
            message,
            {
                "table_headers": table.headers if table else [],
                "face_count": 0,
            },
        )
    except GeminiError:
        # Unreachable ⇒ the table we already read is the best hint.
        faces = table is None
    else:
        faces = _intent_indicates_faces(intent)
    if faces:
        return _faces_branch(image_bytes, message, session)
    return _text_path(image_bytes, message, table)


def process_multiple_images(
    named: list[tuple[str, bytes]], session: WheelSession
) -> ImageResult:
    """Turn multiple uploads into one wheel entry each (Task 13).

    No reading, no detection, no Gemini: the label is the filename minus
    extension, prettified. Each file's bytes are stored in
    ``session.media`` and referenced by its entry's ``image_ref``.
    """
    entries: list[Entry] = []
    for filename, data in named:
        media_id = uuid4().hex
        session.media[media_id] = data
        entries.append(
            Entry(
                id=uuid4().hex,
                label=_prettify_filename(filename),
                image_ref=media_id,
            )
        )
    return ImageResult(entries=entries, table=None, warnings=[])


def _prettify_filename(filename: str) -> str:
    """Filename minus extension, prettified — byte-pinned rule:

    ``re.sub(r"[-_]+", " ", stem).strip().title()`` — e.g. ``Chi.PNG`` →
    ``Chi``, ``team photo.png`` → ``Team Photo``, ``student_a.png`` →
    ``Student A``.
    """
    return re.sub(r"[-_]+", " ", PurePath(filename).stem).strip().title()


def _text_path(
    image_bytes: bytes, message: str, table: Table | None
) -> ImageResult:
    """Text path: a reliable table goes to the picker, else vision."""
    if table is not None and table.columns and table.confidence >= MIN_CONFIDENCE:
        return _from_table(table, message)

    # Reading unreliable — no table, confidence < 60, or no columns
    # (spec §7 step 5) — escalate to Gemini vision.
    try:
        labels = gemini.ask_vision(image_bytes, message)
    except GeminiError:
        if table is not None and table.columns:
            # Vision down but OCR read something usable: proceed
            # best-effort through the picker; its warning travels, or
            # the AI-unreachable ⚠️ when the picker needed no fallback
            # (spec: vision unavailable ⇒ best-effort with ⚠️).
            result = _from_table(table, message)
            if not result.warnings:
                result.warnings = [_AI_UNREACHABLE_WARNING]
            return result
        raise UserFacingError(_UNREADABLE)
    return ImageResult(entries=_entries(labels), table=None, warnings=[])


def _from_table(table: Table, message: str) -> ImageResult:
    """Reliable reading: the picker chooses a column; its warning travels.

    A column with no non-blank values is not a wheel — the reading is
    treated as unreadable so the user gets that copy instead of an
    empty wheel (audit #3)."""
    choice = pick_column(table, message)
    if not any(value.strip() for value in choice.values):
        raise UserFacingError(_UNREADABLE)
    warnings = [choice.warning] if choice.warning else []
    return ImageResult(
        entries=_entries(choice.values), table=table, warnings=warnings
    )


def _faces_branch(
    image_bytes: bytes,
    message: str,
    session: WheelSession,
    boxes: list[tuple[int, int, int, int]] | None = None,
) -> ImageResult:
    """Faces path (spec §7): Haar boxes → crops in ``session.media``.

    Zero detections escalate exactly like the text path's unreliable
    reading: Gemini vision labels the image as plain entries, and a dead
    vision call surfaces the unreadable-image copy.
    """
    if boxes is None:
        boxes = detect_face_boxes(image_bytes)
    if not boxes:
        try:
            labels = gemini.ask_vision(image_bytes, message)
        except GeminiError:
            raise UserFacingError(_UNREADABLE)
        return ImageResult(entries=_entries(labels), table=None, warnings=[])

    entries: list[Entry] = []
    for index, crop in enumerate(crop_faces(image_bytes, boxes)):
        media_id = uuid4().hex
        session.media[media_id] = crop
        entries.append(
            Entry(id=uuid4().hex, label=f"Face {index + 1}", image_ref=media_id)
        )
    return ImageResult(entries=entries, table=None, warnings=[])


def _intent_indicates_faces(intent: Intent) -> bool:
    """Does Gemini's tie-break reply pick the faces branch?

    ``ask_intent``'s frozen ``Intent`` schema carries no faces flag, so a
    faces pick can only surface as face-steer language in the free text
    it returns (question/entries/labels); anything else takes the text
    path (brief: "if it indicates faces … else text").
    """
    reply = " ".join(
        [intent.question or "", *(intent.entries or []), *(intent.labels or [])]
    ).lower()
    return bool(set(reply.split()) & FACE_STEER)


def _entries(labels: list[str]) -> list[Entry]:
    """One entry per non-blank label (mirrors the text-list branch)."""
    entries: list[Entry] = []
    for label in labels:
        stripped = label.strip()
        if stripped:
            entries.append(Entry(id=uuid4().hex, label=stripped))
    return entries
