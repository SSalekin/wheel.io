"""Wheel operations. The server picks the random winner; the browser only animates."""

import random
from uuid import uuid4

from app.models import Entry, WheelSession
from app.services.column_picker import pick_column

# Byte-exact user-facing copy (never vary).
_NO_TABLE_WARNING = "⚠️ No table to switch — upload a list image first."


def spin(
    session: WheelSession, rng: random.Random | random.SystemRandom | None = None
) -> tuple[Entry, int] | None:
    """Pick a uniform winner from ``session.entries``.

    Returns ``(winner, pre_spin_index)`` — the winner and its index before any
    removal — or ``None`` when the wheel is empty. When
    ``session.remove_picked`` is set, the entry is moved from ``entries`` to
    ``picked``.
    """
    if not session.entries:
        return None

    rng = rng or random.SystemRandom()
    pre_spin_index = rng.randrange(len(session.entries))
    winner = session.entries[pre_spin_index]
    if session.remove_picked:
        session.entries.pop(pre_spin_index)
        session.picked.append(winner)
    return winner, pre_spin_index


def wheel_data(session: WheelSession) -> list[dict]:
    """Build the {label, image_url} payload the wheel pane ships to the browser."""
    return [
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


def normalize_labels(items: list[str]) -> list[str]:
    """Clean a batch of labels: strip, drop blanks, dedupe in order.

    Every path that puts labels on the wheel — typed lists, add, the
    table switch — runs its labels through here, so a blank segment or
    a name twice in one message never becomes a wheel entry.
    """
    clean: list[str] = []
    for item in items:
        label = item.strip()
        if label and label not in clean:
            clean.append(label)
    return clean


def replace_entries(
    session: WheelSession, items: list[str] | list[Entry]
) -> None:
    """Replace the wheel wholesale — one entry per item, picks cleared.

    Plain labels are normalized (stripped, blanks dropped, duplicates
    collapsed); prebuilt ``Entry`` objects — image uploads carrying
    ``image_ref`` (Task 13) — are installed as-is, so the media refs
    survive the single replace path. Media that nothing references
    anymore — superseded image crops, since picks are cleared here too —
    is pruned from ``session.media``.
    ``source_table`` is deliberately untouched: ``switch_column`` reuses
    this to swap to another column of the same table, while callers that
    build a fresh wheel from typed text drop the table themselves.
    """
    labels = normalize_labels([item for item in items if isinstance(item, str)])
    built = [item for item in items if isinstance(item, Entry)]
    session.entries = [
        Entry(id=uuid4().hex, label=label) for label in labels
    ] + built
    session.picked = []
    keep = {entry.image_ref for entry in session.entries if entry.image_ref}
    session.media = {
        media_id: data
        for media_id, data in session.media.items()
        if media_id in keep
    }


def add_entries(session: WheelSession, labels: list[str]) -> int:
    """Append ``labels`` to the wheel; the source table is kept (spec §8).

    The batch is normalized first, and the number added is returned —
    "Added N entries." must count what landed, not what was typed.
    """
    clean = normalize_labels(labels)
    session.entries.extend(Entry(id=uuid4().hex, label=label) for label in clean)
    return len(clean)


def remove_label(session: WheelSession, label: str) -> tuple[Entry | None, str | None]:
    """Remove one matching entry — never both duplicates (spec §8).

    Case-insensitive exact match removes the first occurrence, warning
    when identical labels remain behind; otherwise a unique case-insensitive
    substring match removes the single entry containing ``label``;
    otherwise ``(None, exact not-found copy)``.
    """
    if not label:
        # An empty query would "substring-match" every entry.
        return None, f'Couldn\'t find "{label}" on the wheel.'

    lowered = label.lower()
    index = next(
        (
            i
            for i, entry in enumerate(session.entries)
            if entry.label.lower() == lowered
        ),
        None,
    )
    if index is None:
        matches = [
            i
            for i, entry in enumerate(session.entries)
            if lowered in entry.label.lower()
        ]
        if len(matches) != 1:
            return None, f'Couldn\'t find "{label}" on the wheel.'
        index = matches[0]

    removed = session.entries.pop(index)
    duplicates = sum(
        1 for entry in session.entries if entry.label == removed.label
    )
    if duplicates:
        return removed, f'{duplicates} more "{removed.label}" still on the wheel.'
    return removed, None


def switch_column(session: WheelSession, column_query: str) -> str | None:
    """Switch the wheel to another column of the uploaded table (spec §8).

    Returns ``None`` on success — carrying the picker's own warning, if
    any — or the exact no-table copy when there is no table to switch.
    """
    if session.source_table is None:
        return _NO_TABLE_WARNING
    choice = pick_column(session.source_table, column_query)
    replace_entries(session, choice.values)
    return choice.warning


def reset(session: WheelSession) -> None:
    """Clear the wheel and everything derived from it (spec §8)."""
    session.entries = []
    session.picked = []
    session.warnings = []
    session.source_table = None
    session.media = {}
