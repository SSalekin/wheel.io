"""Single-image pipeline: read the table, steer, pick or escalate.

Spec §7: the table reader runs first, then the message steers text vs
faces (``TEXT_STEER``/``FACE_STEER``) with Gemini breaking ambiguous
ties from detector summaries — it never sees the image. The text path
escalates to Gemini vision whenever the reading itself is unreliable;
a reliable table goes through the column picker instead.
"""

from uuid import uuid4

from pydantic import BaseModel

from app.models import Entry, Intent, Table, WheelSession
from app.services import gemini
from app.services.column_picker import pick_column
from app.services.errors import UserFacingError
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

    ``session`` travels along for the faces branch — Task 11 stores face
    crops in ``session.media`` there; Task 10's placeholder never
    reaches it.
    """
    table = read_table(image_bytes)

    tokens = set(message.lower().split())
    text_steer = bool(tokens & TEXT_STEER)
    face_steer = bool(tokens & FACE_STEER)

    if face_steer and not text_steer:
        return _faces_branch()
    if text_steer and not face_steer:
        return _text_path(image_bytes, message, table)

    # Ambiguous steer (both or neither): Gemini picks the path from the
    # message plus detector summaries (spec §7).
    try:
        intent = gemini.ask_intent(
            message,
            {
                "table_headers": table.headers if table else [],
                # Face detection lands with Task 11's faces branch, so
                # nothing has counted faces before the steer runs.
                "face_count": 0,
            },
        )
    except GeminiError:
        # Unreachable ⇒ the table we already read is the best hint.
        faces = table is None
    else:
        faces = _intent_indicates_faces(intent)
    if faces:
        return _faces_branch()
    return _text_path(image_bytes, message, table)


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
    """Reliable reading: the picker chooses a column; its warning travels."""
    choice = pick_column(table, message)
    warnings = [choice.warning] if choice.warning else []
    return ImageResult(
        entries=_entries(choice.values), table=table, warnings=warnings
    )


def _faces_branch() -> ImageResult:
    """Faces path — Task 10 placeholder, no detection yet.

    Task 11 replaces the raise below with real detection (Haar boxes →
    face crops stored in ``session.media`` → ``Face N`` entries).
    """
    raise UserFacingError(_UNREADABLE)


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
