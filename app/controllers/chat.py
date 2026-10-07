"""Chat controller: interpret the message, mutate the session, render partials."""

from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.models import Entry
from app.services.intent_router import interpret

router = APIRouter()

SESSION_EXPIRED = "Session expired — refresh the page to start over."


def _render(request: Request, template: str, context: dict) -> str:
    """Render one HTML fragment through the app's shared templates instance."""
    response = request.app.state.templates.TemplateResponse(request, template, context)
    return response.body.decode()


def _bubble(request: Request, role: str, text: str) -> str:
    return _render(request, "partials/chat_bubble.html", {"role": role, "text": text})


@router.post("/chat", response_class=HTMLResponse)
def chat(
    request: Request,
    token: str = Form(...),
    message: str = Form(""),
    files: list[UploadFile] = File(default=[]),
):
    session = request.app.state.store.get(token)
    if session is None:
        # Stale/unknown token (refresh race): friendly bubble, HTTP 200,
        # no session created. Task 5 extracts this guard into a helper.
        return HTMLResponse(_bubble(request, "system", SESSION_EXPIRED))

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

        wheel_data = [
            {
                "label": entry.label,
                "image_url": (
                    f"/media/{session.token}/{entry.image_ref}"
                    if entry.image_ref
                    else None
                ),
            }
            for entry in session.entries
        ]
        html = _bubble(request, "user", message)
        html += _render(request, "partials/wheel_pane.html", {"wheel_data": wheel_data})
        html += _render(
            request, "partials/picked_list.html", {"picked": session.picked}
        )
        html += _render(
            request,
            "partials/go_form.html",
            {"entries": session.entries, "remove_picked": session.remove_picked},
        )
        return HTMLResponse(html)

    if intent.action == "clarify":
        return HTMLResponse(_bubble(request, "system", intent.question or ""))

    # Follow-up/image actions get their dispatch branches in Tasks 10/12/13;
    # until then the message is echoed without changing state.
    return HTMLResponse(_bubble(request, "user", message))
