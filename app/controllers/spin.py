"""Spin controller: the server picks the winner, the browser only animates."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from app.controllers import bubble, get_session_or_bubble, render_partial
from app.services import wheel_ops

router = APIRouter()

EMPTY_WHEEL = "The wheel is empty — add some entries first."


def _empty_wheel_response(request: Request, session) -> HTMLResponse:
    """Friendly bubble (lands in #banner) plus the regions that can drift:
    Go stays disabled, the picked sidebar stays in sync."""
    return HTMLResponse(
        bubble(request, "system", EMPTY_WHEEL)
        + render_partial(
            request,
            "partials/go_form.html",
            {
                "session": session,
                "entries": session.entries,
                "remove_picked": session.remove_picked,
            },
        )
        + render_partial(
            request, "partials/picked_list.html", {"picked": session.picked, "session": session}
        )
    )


@router.get("/picked", response_class=HTMLResponse)
def picked(request: Request, token: str) -> HTMLResponse:
    """The picked sidebar on its own — wheel.js fetches this when a spin
    animation ends, so the winner lands in the sidebar only after the
    wheel stops (the spin responses no longer ship the list OOB)."""
    session, stale_bubble = get_session_or_bubble(request, token)
    if session is None:
        return stale_bubble
    return HTMLResponse(
        render_partial(
            request,
            "partials/picked_list.html",
            {"picked": session.picked, "session": session},
        )
    )


@router.get("/banner", response_class=HTMLResponse)
def banner(request: Request, token: str) -> HTMLResponse:
    """The winner announcement on its own — wheel.js fetches this when a
    spin animation ends, so the banner lands in #banner only after the
    wheel stops (the spin responses no longer ship it: it used to arrive
    within milliseconds and give the answer away)."""
    session, stale_bubble = get_session_or_bubble(request, token)
    if session is None:
        return stale_bubble
    if session.pending_banner is None:
        return HTMLResponse("")
    return HTMLResponse(
        render_partial(
            request, "partials/banner.html", {"label": session.pending_banner}
        )
    )


@router.post("/spin", response_class=HTMLResponse)
def spin(request: Request, token: str = Form(...)) -> HTMLResponse:
    session, stale_bubble = get_session_or_bubble(request, token)
    if session is None:
        return stale_bubble

    # The whole mutate→render sequence under the session's own lock: a
    # concurrent chat or spin waits its turn instead of interleaving.
    with request.app.state.store.lock_for(token):
        if not session.entries:
            return _empty_wheel_response(request, session)

        # Snapshot the wheel BEFORE the spin: with remove_picked on, the winner
        # is popped by the time we render, and the animation must still be able
        # to land on it (audit fix — same contract as the chat-spin path).
        spin_snapshot = wheel_ops.wheel_data(session)
        result = wheel_ops.spin(session)
        if result is None:  # entries were non-empty above; guard against races
            return _empty_wheel_response(request, session)
        winner, pre_spin_index = result

        # The announcement rides the session, not this response: wheel.js
        # fetches GET /banner when the wheel stops.
        session.pending_banner = winner.label

        # The glow marks the winner in the SHIPPED wheel — only when the winner
        # still has a segment there.
        glow_index = (
            pre_spin_index if any(e is winner for e in session.entries) else None
        )
        html = render_partial(
            request,
            "partials/wheel_pane.html",
            {
                "wheel_data": wheel_ops.wheel_data(session),
                "animate": True,
                "pre_index": pre_spin_index,
                "spin_data": spin_snapshot,
                "glow_index": glow_index,
            },
        )
        html += render_partial(
            request,
            "partials/go_form.html",
            {
                "session": session,
                "entries": session.entries,
                "remove_picked": session.remove_picked,
            },
        )
        return HTMLResponse(html)
