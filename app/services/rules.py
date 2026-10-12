"""Rule-based intent classification for chat messages.

Pure functions only: no I/O and no side effects, so the module stays safe
to import anywhere. ``classify_message`` returns ``None`` when the rules
cannot classify a message confidently — later that means "escalate to
Gemini".
"""

import re

from app.models import Intent

# Instruction verbs/patterns: used to detect a create request in the raw
# message's lead (``_has_create_lead``). Item extraction strips
# instruction language positionally (see _strip_instruction_prefix), never
# by scanning item text.
_INSTRUCTION_RE = re.compile(
    r"\b(?:wheel|make|create|build|pick|choose|random|generate)\b"
    r"|\blist of\b"
    r"|(?:these|this)\s*[:：]",
    re.IGNORECASE,
)

# Words that can appear inside an instruction phrase ("make me a wheel with
# …", "pick the names from …"). Used only to recognize a leading
# instruction PREFIX — an instruction word anywhere in an item is content,
# not an instruction ("Random Guy" is an entry).
_INSTRUCTION_TOKEN_RE = re.compile(
    r"\b(?:wheel|list|make|create|build|pick|choose|random|generate"
    r"|with|for|from|of|me|my|a|an|the|these|this|those"
    r"|names?|students?|entries|items|please)\b",
    re.IGNORECASE,
)

# Tokens that prove a leading run of instruction words really is
# instruction language — a connective or the instruction's object — rather
# than content glued to the first item: "with" makes "make me a wheel with
# Ana" instructions; a bare glued verb ("wheel Ana") is ambiguous.
_STRUCTURAL_TOKENS = frozenset(
    {"with", "for", "from", "of", "these", "this", "list",
     "names", "name", "students", "student", "entries", "items"}
)

# Verb-class instruction tokens. Once a structural token has flagged the
# leading run as instruction language, one of these ENDS the run: it is
# the start of an entry whose own name carries an instruction word
# ("make a wheel with Pick Me" keeps "Pick Me"; "…with Random Guy" keeps
# "Random Guy" whole). Verbs *before* the structural token are ordinary
# instruction language ("pick a random name from …").
_VERB_TOKENS = frozenset(
    {"wheel", "make", "create", "build", "pick", "choose", "random", "generate"}
)

# Articles/filler that may appear inside an instruction phrase. After a
# connective they survive only when they head into the instruction's
# object: "with the names" keeps going, "with The Rock" ends the run
# before "The".
_ARTICLE_TOKENS = frozenset(
    {"the", "a", "an", "this", "these", "those", "my", "your", "our", "me"}
)
_OBJECT_TOKENS = frozenset(
    {"names", "name", "students", "student", "entries", "items", "list"}
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

    Splits on newlines then commas, drops instruction language before a
    line's first ``:``/``：`` (a colon inside data — "9:30", "Bob: red
    team", a URL — survives whole), strips bullet/numbering markers, and
    drops empties. An unambiguous instruction run fused to the first item
    is stripped separately (see ``_strip_instruction_prefix``). When
    nothing list-like survives, the whole message is kept as a single
    item so a plain sentence is never split apart.
    """
    segments: list[str] = []
    for line in text.splitlines():
        for segment in _strip_line_prefix(line).split(","):
            segment = _LEADING_MARKER_RE.sub("", segment).strip()
            if segment:
                segments.append(segment)

    # Instruction language precedes the first item ("make me a wheel with
    # Ana, Bob"), so strip it there — and only there. Anywhere else an
    # instruction word is content: "Random Guy" is an entry, not an
    # instruction.
    if segments:
        first = _strip_instruction_prefix(segments[0])
        segments = ([first] if first else []) + segments[1:]

    if segments:
        return segments
    stripped = text.strip()
    return [stripped] if stripped else []


def _strip_line_prefix(line: str) -> str:
    """Drop instruction language before a line's first colon.

    A colon inside data — "9:30", "Bob: red team", "https://x.com" —
    is content, so the text before it survives unless it carries an
    instruction verb ("wheel: …") or the colon closes the line ("make
    with these:"). An empty prefix loses nothing and is dropped.
    """
    parts = _COLON_RE.split(line, maxsplit=1)
    if len(parts) == 1:
        return line
    prefix, rest = parts
    if not prefix.strip() or not rest.strip():
        return rest
    if any(word.lower() in _VERB_TOKENS for word in prefix.split()):
        return rest
    return line


def _strip_instruction_prefix(segment: str) -> str:
    """Drop a leading instruction phrase from the first item's segment.

    The leading run of instruction words is stripped only when it is
    unambiguous instruction language: it contains a connective or the
    instruction's object (``with``/``for``/``from``/``of``/``these``/
    ``names``/…), or the run spans the whole segment ("make a wheel with
    the names" on its own line carries no entry). "make me a wheel with
    Ana" becomes "Ana"; a lone glued verb ("wheel Ana", "Random Guy",
    "Pick Me") is ambiguous and stays whole — a visible wrong entry
    beats an invisible missing one.

    After a connective the run stops at the first verb-class token (an
    entry's own name starts there: "with Pick Me", "with Random Guy")
    and at an article not leading into the instruction's object ("with
    The Rock"). The trade: adjectival "random" after a connective
    ("with random names") stays with its entry rather than being
    stripped.
    """
    words = segment.split()
    run = 0
    structural = False
    i = 0
    while i < len(words):
        if not _INSTRUCTION_TOKEN_RE.fullmatch(words[i]):
            break  # content starts here
        token = words[i].lower()
        if structural:
            if token in _VERB_TOKENS:
                break  # the entry's own instruction word
            if token in _ARTICLE_TOKENS:
                if i + 1 < len(words) and words[i + 1].lower() in _OBJECT_TOKENS:
                    i += 2  # article + the instruction's object
                    run = i
                    continue
                break  # article straying into an entry ("with The Rock")
        if token in _STRUCTURAL_TOKENS:
            structural = True
        i += 1
        run = i
    if not structural:
        return segment
    return " ".join(words[run:]).strip()


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
    if not _has_create_lead(message):
        return None
    return Intent(action="create_text", entries=items)


def _has_create_lead(message: str) -> bool:
    """Does the message announce list-shaped create intent?

    A create request says so in its LEAD: an instruction verb before the
    first list separator ("wheel:", "make a wheel with these:"), or a
    bare multi-line list (no verb needed). An instruction word appearing
    only later is prose ("hey, take Bob off the wheel"). A followup verb
    at the helm ("add\nSam\nTia") is never a create request.
    """
    if _FOLLOWUP_LED_RE.match(message):
        return False
    if _INSTRUCTION_RE.search(_message_lead(message)):
        return True
    return sum(1 for line in message.splitlines() if line.strip()) >= 2


def _message_lead(message: str) -> str:
    """The message up to its first list separator (newline or comma)."""
    return re.split(r"[\n,]", message, maxsplit=1)[0]


def looks_like_list(items: list[str]) -> bool:
    """True when 2+ items survive and none carries an instruction word.

    The degrade path's best guess for "make a wheel from this": prose
    with an instruction word in it ("hey, take Ana off the wheel") is a
    clause, not a list, so it clarifies instead of building a junk wheel.
    """
    return len(items) >= 2 and not any(_INSTRUCTION_RE.search(item) for item in items)


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
