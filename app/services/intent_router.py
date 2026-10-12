"""Route a chat message to an ``Intent``.

Rules first; when they cannot classify, escalate to Gemini. Any Gemini
failure — unreachable, bad reply, or a reply without usable entries —
degrades to a best-guess fallback plus the exact warning. The signature
below is the frozen contract with the chat controller.
"""

from app.models import Intent, WheelSession
from app.services import gemini
from app.services.gemini import GeminiError
from app.services.rules import classify_message, looks_like_list, parse_items

DEFAULT_CLARIFY = "Could you send me a list or an image to put on the wheel?"
_DEGRADED_WARNING = "⚠️ AI unreachable — best-guess parsing."


def interpret(
    message: str, image_count: int, session: WheelSession
) -> tuple[Intent, list[str]]:
    """Classify a message: rules fast path → Gemini → degrade.

    Returns the intent plus any warnings — empty unless Gemini was
    unreachable or its reply was unusable, in which case a best-guess
    fallback travels with the degradation warning.
    """
    intent = classify_message(message, image_count)
    if intent is not None:
        return intent, []

    state_summary = {
        "entries": [entry.label for entry in session.entries],
        "picked_count": len(session.picked),
        "has_table": session.source_table is not None,
        "table_headers": session.source_table.headers if session.source_table else [],
        "remove_picked": session.remove_picked,
    }
    try:
        intent = gemini.ask_intent(message, state_summary)
    except GeminiError:
        return _degrade(message, image_count)
    # A create without entries would silently wipe the wheel — unusable.
    if intent.action in ("create_text", "add") and not intent.entries:
        return _degrade(message, image_count)
    return intent, []


def _degrade(message: str, image_count: int) -> tuple[Intent, list[str]]:
    """Best-guess fallback for both failure paths, with the exact warning.

    The create fallback needs a clean list (``looks_like_list``): prose
    with an instruction word in it ("hey, take Ana off the wheel") must
    clarify rather than guess a junk wheel.
    """
    items = parse_items(message)
    if looks_like_list(items):
        intent = Intent(action="create_text", entries=items)
    elif image_count >= 2:
        intent = Intent(action="create_images")
    elif image_count == 1:
        intent = Intent("create_image")
    else:
        intent = Intent(action="clarify", question=DEFAULT_CLARIFY)
    return intent, [_DEGRADED_WARNING]
