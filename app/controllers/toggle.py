"""Toggle controller: flip the 'remove picked from wheel' checkbox state."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from app.controllers import get_session_or_bubble

router = APIRouter()


@router.post("/toggle", response_class=HTMLResponse)
def toggle(
    request: Request,
    token: str = Form(...),
    remove_picked: bool = Form(False),
):
    # The checkbox posts its own state with hx-swap="none"; the endpoint flips
    # the stored flag so each change event toggles it exactly once.
    session, stale_bubble = get_session_or_bubble(request, token)
    if session is None:
        return stale_bubble

    session.remove_picked = not session.remove_picked
    return HTMLResponse("")
