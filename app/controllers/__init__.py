"""Shared controller helpers: template rendering and the session-lookup guard."""

from fastapi import Request
from fastapi.responses import HTMLResponse

from app.models.session import WheelSession

SESSION_EXPIRED = "Session expired — refresh the page to start over."


def render_partial(request: Request, template: str, context: dict) -> str:
    """Render one HTML fragment through the app's shared templates instance."""
    response = request.app.state.templates.TemplateResponse(request, template, context)
    return response.body.decode()


def bubble(request: Request, role: str, text: str) -> str:
    return render_partial(request, "partials/chat_bubble.html", {"role": role, "text": text})


def get_session_or_bubble(
    request: Request, token: str
) -> tuple[WheelSession | None, HTMLResponse | None]:
    """Look up the session for ``token``; on a stale/unknown token (refresh
    race) return a friendly bubble instead — HTTP 200, no session created.

    Returns ``(session, None)`` when found, ``(None, bubble_response)``
    otherwise; callers return the bubble response as-is.
    """
    session = request.app.state.store.get(token)
    if session is None:
        return None, HTMLResponse(bubble(request, "system", SESSION_EXPIRED))
    return session, None
