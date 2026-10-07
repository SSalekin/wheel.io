"""Chat controller: interpret the message, mutate the session, render partials."""

from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.controllers import bubble, get_session_or_bubble, render_partial
from app.models import Entry
from app.services import wheel_ops
from app.services.intent_router import interpret

router = APIRouter()


@router.post("/chat", response_class=HTMLResponse)
def chat(
    request: Request,
    token: str = Form(...),
    message: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    session, stale_bubble = get_session_or_bubble(request, token)
    if session is None:
        # Stale/unknown token (refresh race): friendly bubble, HTTP 200,
        # no session created.
        return stale_bubble

    intent, _warnings = interpret(message, len(files), session)

    if intent.action == "create_text":
        # wheel_ops-inline: replace entries (drop empties, dedup), clear
        # picks, drop the table. Task 12 reroutes this via replace_entries.
        labels: list[str] = []
        for item in intent.entries or []:
            label = item.strip()
            if label and label not in labels:
                labels.append(label)
        session.entries = [Entry(id=uuid4().hex, label=label) for label in labels]
        session.picked = []
        session.source_table = None

        html = bubble(request, "user", message)
        html += render_partial(
            request, "partials/wheel_pane.html", {"wheel_data": wheel_ops.wheel_data(session)}
        )
        html += render_partial(
            request, "partials/picked_list.html", {"picked": session.picked}
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

    if intent.action == "clarify":
        return HTMLResponse(bubble(request, "system", intent.question or ""))

    # Follow-up/image actions get their dispatch branches in Tasks 10/12/13;
    # until then the message is echoed without changing state.
    return HTMLResponse(bubble(request, "user", message))
