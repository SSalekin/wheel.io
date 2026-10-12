"""Chat controller: interpret the message, mutate the session, render partials."""

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.controllers import bubble, get_session_or_bubble, render_partial
from app.controllers.spin import EMPTY_WHEEL
from app.models import WheelSession
from app.services import wheel_ops
from app.services.errors import UserFacingError
from app.services.images import process_multiple_images, process_single_image
from app.services.intent_router import interpret
from app.services.mime import is_supported_image

router = APIRouter()

# Upload contract (spec §7): png/jpeg/webp, at most 10 MB, at most 20 files.
_ALLOWED_UPLOAD_TYPES = {"image/png", "image/jpeg", "image/webp"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_MAX_UPLOAD_FILES = 20
_UPLOAD_CHUNK_BYTES = 64 * 1024
_UPLOAD_ERROR = "That file is too big / unsupported — png, jpeg, or webp up to 10 MB."


def _read_validated(files: list[UploadFile]) -> list[bytes]:
    """Read every upload in bounded chunks, refusing anything unsupported.

    Validation lives in the controller, before any dispatch (Task 10
    ruling); every refusal carries the exact copy so the bubble that
    catches it is byte-stable. The declared content-type is checked, and
    so are the bytes themselves: an oversize body is refused one chunk
    past the cap (never fully buffered), and a non-image declared
    ``image/png`` is refused by magic-byte sniffing, not by its label.
    """
    if len(files) > _MAX_UPLOAD_FILES:
        raise UserFacingError(_UPLOAD_ERROR)
    payloads: list[bytes] = []
    for upload in files:
        if upload.content_type not in _ALLOWED_UPLOAD_TYPES:
            raise UserFacingError(_UPLOAD_ERROR)
        data = bytearray()
        while True:
            chunk = upload.file.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            data += chunk
            if len(data) > _MAX_UPLOAD_BYTES:
                raise UserFacingError(_UPLOAD_ERROR)
        if not is_supported_image(bytes(data)):
            raise UserFacingError(_UPLOAD_ERROR)
        payloads.append(bytes(data))
    return payloads


def _wheel_regions(
    request: Request,
    session: WheelSession,
    *,
    animate: bool = False,
    pre_index: int | None = None,
    spin_snapshot: list[dict] | None = None,
    glow_index: int | None = None,
) -> str:
    """The regions that drift after any change to entries or picks.

    Wheel pane + picked sidebar + Go form, each shipped OOB so the wheel
    pane, sidebar, and controls stay in sync with the session.
    ``animate``/``pre_index`` mark the pane for wheel.js when a spin just
    happened (chat-spin, same shape as POST /spin). An animated response
    omits the picked sidebar: wheel.js fetches GET /picked when the spin
    ends, so the winner lands in the sidebar only once the wheel stops.

    The animated pane ships three consistent pieces (audit fix):
    ``spin_snapshot`` — the entries as the spin STARTED (the client animates
    against these, so the pointer lands on the winner even when
    remove_picked has already popped it), ``pre_index`` — the winner's
    index into that snapshot — and ``glow_index``, the winner's index into
    the SHIPPED entries, present only when the winner is still on the
    wheel. A removed winner has no segment left to glow.
    """
    html = render_partial(
        request,
        "partials/wheel_pane.html",
        {
            "wheel_data": wheel_ops.wheel_data(session),
            "animate": animate,
            "pre_index": pre_index,
            "spin_data": spin_snapshot or [],
            "glow_index": glow_index,
        },
    )
    if not animate:
        html += render_partial(
            request, "partials/picked_list.html", {"picked": session.picked, "session": session}
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

    # One mutate→render sequence per request, held whole under the
    # session's own lock: a concurrent chat, spin, or toggle on this
    # session waits its turn instead of interleaving mid-mutation.
    with request.app.state.store.lock_for(token):
        if intent.action == "create_text":
            # The single replace path (T4↔T12 ruling): replace_entries
            # normalizes the parsed list (drop blanks, dedupe). A typed list
            # carries no table.
            wheel_ops.replace_entries(session, intent.entries or [])
            session.source_table = None

            html = bubble(request, "user", message)
            html += warning_html
            html += _wheel_regions(request, session)
            return HTMLResponse(html)

        if intent.action == "clarify":
            return HTMLResponse(bubble(request, "system", intent.question or "") + warning_html)

        if intent.action == "create_image" and len(files) == 1:
            # Exactly one upload ⇒ the single-image pipeline; the multi-image
            # branch follows below.
            try:
                result = process_single_image(upload_bytes[0], message, session)
            except UserFacingError as exc:
                # Caught locally (T10↔T14 ruling); Task 14 adds the
                # app-level safety net for routes that don't catch.
                return HTMLResponse(bubble(request, "system", exc.user_message))
            # The single replace path (also prunes media superseded by the
            # new entries); the replaced wheel carries its table, if any.
            wheel_ops.replace_entries(session, result.entries)
            session.source_table = result.table
            session.warnings.extend(result.warnings)

            html = bubble(request, "user", message)
            html += warning_html
            if result.warnings:
                html += bubble(request, "system", "\n".join(result.warnings))
            html += _wheel_regions(request, session)
            return HTMLResponse(html)

        if intent.action == "create_images" and len(files) >= 2:
            # ≥2 uploads ⇒ one entry per filename — no OCR, no faces, no
            # Gemini (Task 13). Bytes are already validated above; pair each
            # payload with its upload name, then the single replace path.
            named = [
                (upload.filename or "", data)
                for upload, data in zip(files, upload_bytes)
            ]
            result = process_multiple_images(named, session)
            wheel_ops.replace_entries(session, result.entries)
            # The replaced wheel is filename-backed: drop any table from an
            # earlier upload so `switch` can't rebuild from entries the user
            # no longer sees (final-review fix).
            session.source_table = None

            html = bubble(request, "user", message)
            html += warning_html
            html += _wheel_regions(request, session)
            return HTMLResponse(html)

        if intent.action == "add":
            added_count = wheel_ops.add_entries(session, intent.entries or [])

            html = bubble(request, "user", message)
            html += warning_html
            html += bubble(request, "system", f"Added {added_count} entries.")
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
            # Snapshot the wheel BEFORE the spin: with remove_picked on, the
            # winner is popped by the time we render, and the animation must
            # still be able to land on it.
            spin_snapshot = wheel_ops.wheel_data(session)
            result = wheel_ops.spin(session)
            html = bubble(request, "user", message)
            html += warning_html
            if result is None:
                # Same copy as an empty POST /spin; nothing changed, so no regions.
                return HTMLResponse(html + bubble(request, "system", EMPTY_WHEEL))
            winner, pre_spin_index = result

            # The glow marks the winner in the SHIPPED wheel — only when the
            # winner still has a segment there.
            glow_index = (
                pre_spin_index if any(e is winner for e in session.entries) else None
            )
            html += bubble(request, "system", f"🎉 {winner.label}")
            # Chat responses land in #chat-log, so the #banner region swaps out
            # of band — the banner partial is the region's content.
            html += (
                '<div id="banner" class="banner" hx-swap-oob="true">'
                + render_partial(request, "partials/banner.html", {"label": winner.label})
                + "</div>"
            )
            html += _wheel_regions(
                request,
                session,
                animate=True,
                pre_index=pre_spin_index,
                spin_snapshot=spin_snapshot,
                glow_index=glow_index,
            )
            return HTMLResponse(html)

        # Every rule-bound action has a branch above. What can still land
        # here is a Gemini reply carrying an image action with a mismatched
        # file count (e.g. create_images with nothing attached): echo the
        # message without touching state — never mutate on an unmatched intent.
        return HTMLResponse(bubble(request, "user", message) + warning_html)
