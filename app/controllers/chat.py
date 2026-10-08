"""Chat controller: interpret the message, mutate the session, render partials."""

from uuid import uuid4

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.controllers import bubble, get_session_or_bubble, render_partial
from app.models import Entry
from app.services import wheel_ops
from app.services.errors import UserFacingError
from app.services.images import process_single_image
from app.services.intent_router import interpret

router = APIRouter()

# Upload contract (spec §7): png/jpeg/webp, at most 10 MB.
_ALLOWED_UPLOAD_TYPES = {"image/png", "image/jpeg", "image/webp"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_UPLOAD_ERROR = "That file is too big / unsupported — png, jpeg, or webp up to 10 MB."


def _read_validated(files: list[UploadFile]) -> list[bytes]:
    """Read every upload, refusing unsupported types and oversizes.

    Validation lives in the controller, before any dispatch (Task 10
    ruling); every refusal carries the exact copy so the bubble that
    catches it is byte-stable.
    """
    payloads: list[bytes] = []
    for upload in files:
        data = upload.file.read()
        if (
            upload.content_type not in _ALLOWED_UPLOAD_TYPES
            or len(data) > _MAX_UPLOAD_BYTES
        ):
            raise UserFacingError(_UPLOAD_ERROR)
        payloads.append(data)
    return payloads


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

    # Bad uploads are refused before dispatch (Task 10): HTTP 200 bubble
    # with the exact copy; the app-level handler lands in Task 14.
    try:
        upload_bytes = _read_validated(files)
    except UserFacingError as exc:
        return HTMLResponse(bubble(request, "system", exc.user_message))

    intent, warnings = interpret(message, len(files), session)
    # Degradation notes ride along after the main bubble (Task 7 ruling).
    warning_html = "".join(bubble(request, "system", text) for text in warnings)

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
        html += warning_html
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
        return HTMLResponse(bubble(request, "system", intent.question or "") + warning_html)

    if intent.action == "create_image" and len(files) == 1:
        # Exactly one upload ⇒ the single-image pipeline; multi-image
        # dispatch lands with Task 13.
        try:
            result = process_single_image(upload_bytes[0], message, session)
        except UserFacingError as exc:
            # Caught locally (T10↔T14 ruling); Task 14 adds the
            # app-level safety net for routes that don't catch.
            return HTMLResponse(bubble(request, "system", exc.user_message))
        session.entries = result.entries
        session.source_table = result.table
        session.picked = []
        session.warnings.extend(result.warnings)

        html = bubble(request, "user", message)
        html += warning_html
        if result.warnings:
            html += bubble(request, "system", "\n".join(result.warnings))
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

    # Follow-up/multi-image actions get their dispatch branches in Tasks 12/13;
    # until then the message is echoed without changing state.
    return HTMLResponse(bubble(request, "user", message) + warning_html)
