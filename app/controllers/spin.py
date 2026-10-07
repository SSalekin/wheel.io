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
            request, "partials/picked_list.html", {"picked": session.picked}
        )
    )


@router.post("/spin", response_class=HTMLResponse)
def spin(request: Request, token: str = Form(...)):
    session, stale_bubble = get_session_or_bubble(request, token)
    if session is None:
        return stale_bubble

    if not session.entries:
        return _empty_wheel_response(request, session)

    result = wheel_ops.spin(session)
    if result is None:  # entries were non-empty above; guard against races
        return _empty_wheel_response(request, session)
    winner, pre_spin_index = result

    html = render_partial(request, "partials/banner.html", {"label": winner.label})
    html += render_partial(
        request,
        "partials/wheel_pane.html",
        {
            "wheel_data": wheel_ops.wheel_data(session),
            "animate": True,
            "pre_index": pre_spin_index,
        },
    )
    html += render_partial(request, "partials/picked_list.html", {"picked": session.picked})
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
