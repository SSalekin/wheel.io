"""Wheel operations. The server picks the random winner; the browser only animates."""

import random

from app.models import Entry, WheelSession


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
