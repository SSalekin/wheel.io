"""Chat controller: interpret the message, mutate the session, render partials."""

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.controllers import bubble, get_session_or_bubble, render_partial
from app.controllers.spin import EMPTY_WHEEL
from app.models import WheelSession
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


def _wheel_regions(
    request: Request,
    session: WheelSession,
    *,
    animate: bool = False,
    pre_index: int | None = None,
) -> str:
    """The regions that drift after any change to entries or picks.

    Wheel pane + picked sidebar + Go form, each shipped OOB so the wheel
    pane, sidebar, and controls stay in sync with the session.
    ``animate``/``pre_index`` mark the pane for wheel.js when a spin just
    happened (chat-spin, same shape as POST /spin).
    """
    html = render_partial(
        request,
        "partials/wheel_pane.html",
        {
            "wheel_data": wheel_ops.wheel_data(session),
            "animate": animate,
            "pre_index": pre_index,
        },
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
    return html


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
        # Normalize the parsed list (drop empties, dedup), then hand the
        # replace to wheel_ops — the single replace path (T4↔T12 ruling).
        # A typed list carries no table.
        labels: list[str] = []
        for item in intent.entries or []:
            label = item.strip()
            if label and label not in labels:
                labels.append(label)
        wheel_ops.replace_entries(session, labels)
        session.source_table = None

        html = bubble(request, "user", message)
        html += warning_html
        html += _wheel_regions(request, session)
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
        html += _wheel_regions(request, session)
        return HTMLResponse(html)

    if intent.action == "add":
        added = intent.entries or []
        wheel_ops.add_entries(session, added)

        html = bubble(request, "user", message)
        html += warning_html
        html += bubble(request, "system", f"Added {len(added)} entries.")
        html += _wheel_regions(request, session)
        return HTMLResponse(html)

    if intent.action == "remove":
        html = bubble(request, "user", message)
        html += warning_html
        for label in intent.labels or []:
            entry, note = wheel_ops.remove_label(session, label)
            # Warnings (duplicates left, not-found) arrive in the note slot;
            # a clean removal confirms with the wheel's own label.
            text = note if note is not None else f'Removed "{entry.label}".'
            html += bubble(request, "system", text)
        html += _wheel_regions(request, session)
        return HTMLResponse(html)

    if intent.action == "switch_column":
        note = wheel_ops.switch_column(session, intent.column or "")

        html = bubble(request, "user", message)
        html += warning_html
        text = note or f"Switched — the wheel now shows {len(session.entries)} entries."
        html += bubble(request, "system", text)
        html += _wheel_regions(request, session)
        return HTMLResponse(html)

    if intent.action == "reset":
        wheel_ops.reset(session)

        html = bubble(request, "user", message)
        html += warning_html
        html += bubble(request, "system", "Wheel cleared.")
        html += _wheel_regions(request, session)
        return HTMLResponse(html)

    if intent.action == "spin":
        result = wheel_ops.spin(session)
        html = bubble(request, "user", message)
        html += warning_html
        if result is None:
            # Same copy as an empty POST /spin; nothing changed, so no regions.
            return HTMLResponse(html + bubble(request, "system", EMPTY_WHEEL))
        winner, pre_spin_index = result

        html += bubble(request, "system", f"🎉 {winner.label}")
        # Chat responses land in #chat-log, so the #banner region swaps out
        # of band — the banner partial is the region's content.
        html += (
            '<div id="banner" class="banner" hx-swap-oob="true">'
            + render_partial(request, "partials/banner.html", {"label": winner.label})
            + "</div>"
        )
        html += _wheel_regions(
            request, session, animate=True, pre_index=pre_spin_index
        )
        return HTMLResponse(html)

    # Follow-up dispatch above; multi-image (create_images) gets its branch
    # in Task 13 — until then the message is echoed without changing state.
    return HTMLResponse(bubble(request, "user", message) + warning_html)
