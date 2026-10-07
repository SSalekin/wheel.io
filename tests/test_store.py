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
