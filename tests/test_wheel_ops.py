import random

from app.models import Entry, WheelSession
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
