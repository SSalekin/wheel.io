"""Rule-based intent classification for chat messages.

Pure functions only: no I/O and no side effects, so the module stays safe
to import anywhere. ``classify_message`` returns ``None`` when the rules
cannot classify a message confidently — later that means "escalate to
Gemini".
"""

import re

from app.models import Intent

# Instruction verbs/patterns: used both to detect a create request in the
# raw message and to drop instruction lines while parsing items.
_INSTRUCTION_RE = re.compile(
    r"\b(?:wheel|make|create|build|pick|choose|random|generate)\b"
    r"|\blist of\b"
    r"|(?:these|this)\s*[:：]",
    re.IGNORECASE,
)

_COLON_RE = re.compile(r"[:：]")
_LEADING_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")

# A message led by a follow-up verb (e.g. "add\nSam\nTia") is a follow-up,
# never a create request, even when it parses to a multi-line list.
_FOLLOWUP_LED_RE = re.compile(
    r"\s*(?:remove|switch|reset|clear|spin|go|add)\b", re.IGNORECASE
)

_REMOVE_RE = re.compile(r"\bremove\b\s*(?P<body>.*)", re.IGNORECASE | re.DOTALL)
_SWITCH_RE = re.compile(r"\bswitch\s+to\s+(?P<column>[^\n]+)", re.IGNORECASE)
_RESET_RE = re.compile(r"\b(?:reset|clear)\b", re.IGNORECASE)
_SPIN_RE = re.compile(r"\b(?:spin|go)\b", re.IGNORECASE)
_ADD_RE = re.compile(r"\badd\b\s*(?P<body>.*)", re.IGNORECASE | re.DOTALL)

_SPLIT_RE = re.compile(r"\n|,|\band\b", re.IGNORECASE)

# Column headers that point at tabular/text data (reused by Task 10).
TEXT_STEER: set[str] = {
    "name",
    "names",
    "student",
    "students",
    "id",
    "ids",
    "email",
    "emails",
    "roster",
    "list",
    "table",
    "column",
}

# Terms that point at face/avatar imagery (reused by Task 10).
FACE_STEER: set[str] = {
    "photo",
    "photos",
    "picture",
    "pictures",
    "face",
    "faces",
    "avatar",
    "avatars",
    "selfie",
    "headshot",
}


def parse_items(text: str) -> list[str]:
    """Extract candidate items from a chat message.

    Splits on newlines then commas, drops instruction prefixes (text
    before ``:``/``：`` on a line) and instruction segments, strips
    bullet/numbering markers, and drops empties. When nothing list-like
    survives, the whole message is kept as a single item so a plain
    sentence is never split apart.
    """
    segments: list[str] = []
    for line in text.splitlines():
        after_prefix = re.split(r"[:：]", line, maxsplit=1)[-1]
        for segment in after_prefix.split(","):
            segment = _LEADING_MARKER_RE.sub("", segment).strip()
            if segment:
                segments.append(segment)

    kept = [s for s in segments if not _INSTRUCTION_RE.search(s)]
    if kept:
        return kept
    stripped = text.strip()
    return [stripped] if stripped else []


def classify_message(message: str, image_count: int) -> Intent | None:
    """Classify a chat message (plus attached image count) into an Intent.

    Returns ``None`` when the rules do not apply (smalltalk or anything
    too ambiguous), which later means "escalate to Gemini".
    """
    return (
        _classify_create(message)
        or _classify_followup(message)
        or _classify_image(image_count)
    )


def _classify_create(message: str) -> Intent | None:
    items = parse_items(message)
    if len(items) < 2:
        return None
    if _FOLLOWUP_LED_RE.match(message):
        return None
    has_verb = _INSTRUCTION_RE.search(message) is not None
    has_newline_list = (
        sum(1 for line in message.splitlines() if line.strip()) >= 2
    )
    # Comma-only lists additionally require a verb.
    if not (has_verb or has_newline_list):
        return None
    return Intent(action="create_text", entries=items)


def _classify_followup(message: str) -> Intent | None:
    match = _REMOVE_RE.search(message)
    if match:
        return Intent(action="remove", labels=_split_items(match.group("body")))
    match = _SWITCH_RE.search(message)
    if match:
        return Intent(action="switch_column", column=match.group("column").strip())
    if _RESET_RE.search(message):
        return Intent(action="reset")
    if _SPIN_RE.search(message):
        return Intent(action="spin")
    match = _ADD_RE.search(message)
    if match:
        return Intent(action="add", entries=_split_items(match.group("body")))
    return None


def _classify_image(image_count: int) -> Intent | None:
    if image_count >= 2:
        return Intent(action="create_images")
    if image_count == 1:
        return Intent(action="create_image")
    return None


def _split_items(text: str) -> list[str]:
    parts = _SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]
