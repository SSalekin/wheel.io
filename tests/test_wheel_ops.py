import random

from app.models import Entry, Table, TableColumn, WheelSession
from app.services import wheel_ops


def _session(labels: list[str], remove_picked: bool = False) -> WheelSession:
    return WheelSession(
        token="token",
        created_at=0.0,
        entries=[Entry(id=f"id{i}", label=label) for i, label in enumerate(labels)],
        remove_picked=remove_picked,
    )


def test_spin_moves_entry_when_removal_on():
    session = _session(["Ana", "Bob", "Chi"], remove_picked=True)
    original = list(session.entries)

    result = wheel_ops.spin(session, rng=random.Random(0))

    assert result is not None
    winner, pre_spin_index = result
    assert winner in original
    assert original[pre_spin_index] == winner
    assert len(session.entries) == 2
    assert len(session.picked) == 1
    assert winner in session.picked
    assert winner not in session.entries


def test_spin_keeps_entry_when_removal_off():
    session = _session(["Ana", "Bob", "Chi"], remove_picked=False)
    entries_before = list(session.entries)
    picked_before = list(session.picked)

    result = wheel_ops.spin(session, rng=random.Random(0))

    assert result is not None
    winner, pre_spin_index = result
    assert winner == entries_before[pre_spin_index]
    assert session.entries == entries_before
    assert session.picked == picked_before


def test_spin_empty_returns_none():
    session = _session([])

    assert wheel_ops.spin(session, rng=random.Random(0)) is None


def test_wheel_data_carries_image_url():
    # Thumbnail ruling: entries with an image_ref ship their media URL;
    # plain entries ship null so wheel.js stays label-only.
    session = WheelSession(
        token="token",
        created_at=0.0,
        entries=[
            Entry(id="id0", label="Ana", image_ref="crop1"),
            Entry(id="id1", label="Bob"),
        ],
    )

    assert wheel_ops.wheel_data(session) == [
        {"label": "Ana", "image_url": "/media/token/crop1"},
        {"label": "Bob", "image_url": None},
    ]


def _table() -> Table:
    return Table(
        columns=[
            TableColumn(header="name", values=["Ana", "Bob"]),
            TableColumn(header="id", values=["A1", "B2"]),
        ],
        confidence=95.0,
    )


def test_remove_first_exact_duplicate_warns():
    # Review Focus #5: duplicates ⇒ remove ONE and say so, byte-exact copy.
    session = _session(["Bob", "Bob", "Ana"])

    removed, warning = wheel_ops.remove_label(session, "Bob")

    assert removed is not None
    assert removed.label == "Bob"
    assert [entry.label for entry in session.entries] == ["Bob", "Ana"]
    assert warning == '1 more "Bob" still on the wheel.'


def test_remove_never_removes_both():
    session = _session(["Bob", "Bob", "Ana"])

    wheel_ops.remove_label(session, "Bob")

    assert len(session.entries) == 2
    assert sum(1 for entry in session.entries if entry.label == "Bob") == 1


def test_remove_unique_substring():
    session = _session(["Bob Smith", "Ana"])

    removed, warning = wheel_ops.remove_label(session, "bob")

    assert removed is not None
    assert removed.label == "Bob Smith"
    assert warning is None
    assert [entry.label for entry in session.entries] == ["Ana"]


def test_remove_missing():
    session = _session(["Ana", "Bob"])

    removed, warning = wheel_ops.remove_label(session, "Zed")

    assert removed is None
    assert warning == 'Couldn\'t find "Zed" on the wheel.'
    assert [entry.label for entry in session.entries] == ["Ana", "Bob"]

    # Empty query never substring-matches its way onto the wheel.
    assert wheel_ops.remove_label(session, "") == (
        None,
        'Couldn\'t find "" on the wheel.',
    )
    assert len(session.entries) == 2


def test_add_appends_and_keeps_table():
    session = _session(["Ana"])
    session.source_table = _table()

    wheel_ops.add_entries(session, ["Sam", "Tia"])

    assert [entry.label for entry in session.entries] == ["Ana", "Sam", "Tia"]
    assert session.source_table is not None
    assert session.source_table.headers == ["name", "id"]


def test_switch_without_table_warns():
    session = _session(["Ana"])

    warning = wheel_ops.switch_column(session, "id")

    assert warning == "⚠️ No table to switch — upload a list image first."


def test_switch_replaces_entries_and_clears_picked():
    session = _session(["Ana", "Bob"])
    session.source_table = _table()
    session.picked = [Entry(id="p0", label="Chi")]

    warning = wheel_ops.switch_column(session, "id")

    assert warning is None
    assert [entry.label for entry in session.entries] == ["A1", "B2"]
    assert session.picked == []
    assert session.source_table is not None  # still switchable afterwards


def test_reset_clears_everything():
    session = _session(["Ana"])
    session.picked = [Entry(id="p0", label="Chi")]
    session.warnings = ["⚠️ stale note"]
    session.source_table = _table()
    session.media = {"crop": b"png-bytes"}

    wheel_ops.reset(session)

    assert session.entries == []
    assert session.picked == []
    assert session.warnings == []
    assert session.source_table is None
    assert session.media == {}
