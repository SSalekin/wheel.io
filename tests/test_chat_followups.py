"""Chat dispatch for follow-ups: add / remove / switch / reset / chat-spin.

Each test posts a rule-classified follow-up at the real POST /chat endpoint
and asserts the pinned copy plus the OOB regions that keep the wheel pane,
picked sidebar, and Go form in sync.
"""

import html
import re

import pytest

from app.main import app
from app.models import Entry, Table, TableColumn


@pytest.fixture(autouse=True)
def _no_gemini_key(monkeypatch):
    """Follow-ups classify through rules; never let a stray path hit the
    real Gemini API (mirrors test_chat's fixture)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _page_token(client) -> str:
    """Load GET / and pull the per-page hidden token from the chat form."""
    body = client.get("/").text
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "page must render a hidden token input with a non-empty value"
    return match.group(1)


def _seed_wheel(client, token: str) -> None:
    response = client.post(
        "/chat", data={"token": token, "message": "wheel:\nAna\nBob\nChi"}
    )
    assert response.status_code == 200


def test_chat_remove_flow(client):
    token = _page_token(client)
    _seed_wheel(client, token)

    response = client.post("/chat", data={"token": token, "message": "remove Bob"})

    assert response.status_code == 200
    body = html.unescape(response.text)
    assert 'Removed "Bob".' in body  # confirmation bubble
    assert 'id="wheel-pane"' in response.text  # regions that changed ship OOB
    assert 'id="picked-list"' in response.text
    assert "hx-swap-oob" in response.text
    session = app.state.store.get(token)
    assert [entry.label for entry in session.entries] == ["Ana", "Chi"]

    # Not-found rides the same bubble path — byte-exact copy.
    missing = client.post("/chat", data={"token": token, "message": "remove Zed"})
    assert 'Couldn\'t find "Zed" on the wheel.' in html.unescape(missing.text)
    session = app.state.store.get(token)
    assert [entry.label for entry in session.entries] == ["Ana", "Chi"]


def test_chat_remove_duplicate_warns(client):
    # Ruling 3: the duplicate-remove warning renders byte-exact in chat and
    # only one of the duplicates goes away (Review Focus #5 end to end).
    token = _page_token(client)
    session = app.state.store.get(token)
    session.entries = [
        Entry(id="a", label="Ana"),
        Entry(id="b", label="Bob"),
        Entry(id="c", label="Bob"),
    ]

    response = client.post("/chat", data={"token": token, "message": "remove Bob"})

    assert response.status_code == 200
    assert '1 more "Bob" still on the wheel.' in html.unescape(response.text)
    session = app.state.store.get(token)
    assert [entry.label for entry in session.entries] == ["Ana", "Bob"]


def test_chat_spin_flow(client):
    # Chat-spin carries the same partial set as POST /spin: winner bubble +
    # OOB banner, wheel pane (animate + pre-index), Go form. The picked
    # sidebar is deferred — wheel.js fetches GET /picked when the spin
    # animation ends so the winner appears only once the wheel stops.
    token = _page_token(client)
    _seed_wheel(client, token)

    response = client.post("/chat", data={"token": token, "message": "spin"})

    assert response.status_code == 200
    body = response.text
    assert "🎉" in body
    assert re.search(r'<div class="bubble system">🎉 (Ana|Bob|Chi)</div>', body)
    assert 'id="banner"' not in body  # banner deferred: served by GET /banner once the wheel stops
    assert 'data-animate="1"' in body
    assert re.search(r'data-pre-index="\d+"', body)
    assert 'id="wheel-pane"' in body
    assert 'id="picked-list"' not in body
    assert 'id="go-form"' in body
    assert "hx-swap-oob" in body


def test_chat_spin_empty_wheel_bubbles(client):
    # Empty wheel ⇒ the pinned empty copy (same as POST /spin), no winner.
    token = _page_token(client)

    response = client.post("/chat", data={"token": token, "message": "spin"})

    assert response.status_code == 200
    assert "The wheel is empty — add some entries first." in response.text
    assert "🎉" not in response.text


def test_chat_add_flow(client):
    token = _page_token(client)
    _seed_wheel(client, token)

    response = client.post(
        "/chat", data={"token": token, "message": "add Sam and Tia"}
    )

    assert response.status_code == 200
    assert "Added 2 entries." in html.unescape(response.text)
    assert 'id="wheel-pane"' in response.text
    assert "hx-swap-oob" in response.text
    session = app.state.store.get(token)
    assert [entry.label for entry in session.entries] == [
        "Ana",
        "Bob",
        "Chi",
        "Sam",
        "Tia",
    ]


def test_chat_switch_flow(client):
    token = _page_token(client)
    session = app.state.store.get(token)
    session.entries = [Entry(id="e0", label="Ana")]
    session.picked = [Entry(id="p0", label="Chi")]
    session.source_table = Table(
        columns=[
            TableColumn(header="name", values=["Ana", "Bob"]),
            TableColumn(header="id", values=["A1", "B2"]),
        ],
        confidence=95.0,
    )

    response = client.post("/chat", data={"token": token, "message": "switch to id"})

    assert response.status_code == 200
    assert "Switched — the wheel now shows 2 entries." in html.unescape(
        response.text
    )
    assert 'id="wheel-pane"' in response.text
    assert "hx-swap-oob" in response.text
    session = app.state.store.get(token)
    assert [entry.label for entry in session.entries] == ["A1", "B2"]
    assert session.picked == []
    assert session.source_table is not None  # still switchable afterwards


def test_chat_reset_flow(client):
    token = _page_token(client)
    _seed_wheel(client, token)
    session = app.state.store.get(token)
    session.picked = [Entry(id="p0", label="Chi")]
    session.media = {"crop": b"png-bytes"}
    session.source_table = Table(
        columns=[TableColumn(header="id", values=["A1"])], confidence=95.0
    )

    response = client.post("/chat", data={"token": token, "message": "reset"})

    assert response.status_code == 200
    assert "Wheel cleared." in html.unescape(response.text)
    assert 'id="wheel-pane"' in response.text
    assert "hx-swap-oob" in response.text
    session = app.state.store.get(token)
    assert session.entries == []
    assert session.picked == []
    assert session.source_table is None
    assert session.media == {}


def test_chat_add_dedupes_and_counts_clean_entries(client):
    # "add Sam, Sam, Tia" — the batch collapses before it reaches the
    # wheel, so the count matches what was actually added.
    token = _page_token(client)
    _seed_wheel(client, token)

    response = client.post(
        "/chat", data={"token": token, "message": "add Sam, Sam, Tia"}
    )

    assert response.status_code == 200
    assert "Added 2 entries." in html.unescape(response.text)
    session = app.state.store.get(token)
    assert [entry.label for entry in session.entries] == [
        "Ana",
        "Bob",
        "Chi",
        "Sam",
        "Tia",
    ]
