from app.models.entry import Entry
from app.models.intent import Intent
from app.models.session import WheelSession
from app.models.table import Table, TableColumn


def test_entry_defaults():
    entry = Entry(id="e1", label="Alice")

    assert entry.id == "e1"
    assert entry.label == "Alice"
    assert entry.image_ref is None


def test_intent_optional_fields():
    intent = Intent(action="spin")

    assert intent.action == "spin"
    assert intent.entries is None
    assert intent.labels is None
    assert intent.column is None
    assert intent.question is None


def test_table_headers_and_get_column():
    table = Table(
        columns=[
            TableColumn(header="Name", values=["Alice", "Bob"]),
            TableColumn(header="ID", values=["A1", "A2"]),
        ],
        confidence=0.9,
    )

    assert table.headers == ["Name", "ID"]
    # Exact match on header, not case-insensitive.
    assert table.get_column("id") is None
    assert table.get_column("ID").values == ["A1", "A2"]
    assert table.get_column("Name").values == ["Alice", "Bob"]
    assert table.get_column("Missing") is None


def test_wheel_session_defaults():
    session = WheelSession(token="tok", created_at=123.0)

    assert session.token == "tok"
    assert session.created_at == 123.0
    assert session.entries == []
    assert session.picked == []
    assert session.remove_picked is False
    assert session.source_table is None
    assert session.warnings == []
    assert session.media == {}

    # Mutable defaults must not be shared between instances.
    other = WheelSession(token="tok2", created_at=124.0)
    session.entries.append(Entry(id="e1", label="Alice"))
    session.media["img.png"] = b"bytes"
    assert other.entries == []
    assert other.media == {}
