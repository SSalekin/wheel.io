"""Route a chat message to an ``Intent``.

Rules-only version: ``classify_message`` with a clarify fallback. Task 7
replaces this body with rules → Gemini escalation plus degradation — the
signature below is the frozen contract between the two.
"""

from app.models import Intent, WheelSession
from app.services.rules import classify_message

DEFAULT_CLARIFY = "Could you send me a list or an image to put on the wheel?"


def interpret(
    message: str, image_count: int, session: WheelSession
) -> tuple[Intent, list[str]]:
    """Classify a message; unclassifiable ⇒ clarify with the default question.

    Returns the intent plus any warnings (always empty while rules-only).
    ``session`` is unused for now — Task 7 builds the Gemini state summary
    from it.
    """
    intent = classify_message(message, image_count)
    if intent is None:
        intent = Intent(action="clarify", question=DEFAULT_CLARIFY)
    return intent, []
