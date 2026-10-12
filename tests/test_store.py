from app.models import Entry
from app.state.store import InMemoryStateStore


def test_create_and_get():
    store = InMemoryStateStore()

    session = store.create()

    assert session.token
    assert store.get(session.token) is session


def test_get_unknown_returns_none():
    store = InMemoryStateStore()

    assert store.get("no-such-token") is None


def test_expire_removes_old():
    store = InMemoryStateStore(ttl_seconds=10)
    old = store.create()
    old.created_at -= 100  # Backdate beyond the TTL.
    fresh = store.create()

    deleted = store.expire(now=old.created_at + 11)

    assert deleted == 1
    assert store.get(old.token) is None
    assert store.get(fresh.token) is fresh


def test_expire_returns_count():
    store = InMemoryStateStore(ttl_seconds=10)
    first = store.create()
    second = store.create()
    third = store.create()
    first.created_at -= 100
    second.created_at -= 100

    assert store.expire(now=first.created_at + 11) == 2
    assert store.expire() == 0
    assert store.get(third.token) is third


def test_lock_for_is_stable_per_token():
    store = InMemoryStateStore()

    assert store.lock_for("a") is store.lock_for("a")
    assert store.lock_for("a") is not store.lock_for("b")


def test_expire_drops_session_locks():
    store = InMemoryStateStore(ttl_seconds=10)
    session = store.create()
    lock = store.lock_for(session.token)

    store.expire(now=session.created_at + 11)

    assert store.lock_for(session.token) is not lock


def test_concurrent_session_mutations_never_raise():
    # Sessions are mutated outside the store's dict lock (spin pops an
    # entry while a chat replaces the wheel); the per-session lock keeps
    # each mutate→render sequence whole. Without it, a spin can pop from
    # a wheel another thread just replaced and raise IndexError.
    import threading

    from app.services import wheel_ops

    store = InMemoryStateStore()
    session = store.create()
    session.entries = [Entry(id=f"e{i}", label=f"Name {i}") for i in range(120)]
    session.remove_picked = True
    errors: list[Exception] = []

    def hammer():
        try:
            for _ in range(60):
                with store.lock_for(session.token):
                    wheel_ops.spin(session)
                    wheel_ops.add_entries(session, ["X", "", "X"])
                    if len(session.entries) > 300:
                        wheel_ops.replace_entries(session, ["Ana", "Bob", "Chi"])
        except Exception as exc:  # pragma: no cover - only on race
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
