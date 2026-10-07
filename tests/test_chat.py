import re

import pytest

from app.main import app

SESSION_EXPIRED = "Session expired — refresh the page to start over."
CLARIFY_DEFAULT = "Could you send me a list or an image to put on the wheel?"
DEGRADED_WARNING = "⚠️ AI unreachable — best-guess parsing."


@pytest.fixture(autouse=True)
def _no_gemini_key(monkeypatch):
    """Escalation (Task 7) must degrade locally in tests — never hit the
    real Gemini API, whatever the machine's environment carries."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _page_token(client) -> str:
    """Load GET / and pull the per-page hidden token from the chat form."""
    body = client.get("/").text
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "page must render a hidden token input with a non-empty value"
    return match.group(1)


def test_create_text_builds_wheel(client):
    token = _page_token(client)

    response = client.post(
        "/chat",
        data={"token": token, "message": "wheel:\nAna\nBob\nChi"},
    )

    assert response.status_code == 200

    session = app.state.store.get(token)
    assert session is not None
    assert [entry.label for entry in session.entries] == ["Ana", "Bob", "Chi"]
    assert session.picked == []
    assert session.source_table is None

    body = response.text
    assert "Ana" in body
    assert "hx-swap-oob" in body
    assert 'id="wheel-pane"' in body
    assert 'id="picked-list"' in body
    assert 'id="go-form"' in body


def test_stale_token_returns_friendly_bubble(client):
    response = client.post("/chat", data={"token": "nope", "message": "wheel:\nAna"})

    assert response.status_code == 200
    assert SESSION_EXPIRED in response.text
    assert app.state.store.get("nope") is None


def test_empty_message_no_images_clarifies(client):
    token = _page_token(client)

    response = client.post("/chat", data={"token": token, "message": ""})

    assert response.status_code == 200
    assert CLARIFY_DEFAULT in response.text


def test_page_load_mints_token(client):
    first = _page_token(client)
    second = _page_token(client)

    assert first
    assert second
    assert first != second


def test_degraded_warning_renders_as_system_bubble(client):
    """Carried ruling: interpret()'s warnings must be visible in chat."""
    token = _page_token(client)

    response = client.post(
        "/chat", data={"token": token, "message": "please pick students for me"}
    )

    assert response.status_code == 200
    assert CLARIFY_DEFAULT in response.text
    warning_bubble = f'<div class="bubble system">{DEGRADED_WARNING}</div>'
    assert warning_bubble in response.text


def test_degraded_warning_renders_on_create_text(client):
    """Degradation can still produce create_text — the note rides along."""
    token = _page_token(client)

    response = client.post("/chat", data={"token": token, "message": "Ana, Bob"})

    assert response.status_code == 200
    assert "hx-swap-oob" in response.text
    warning_bubble = f'<div class="bubble system">{DEGRADED_WARNING}</div>'
    assert warning_bubble in response.text
    session = app.state.store.get(token)
    assert [entry.label for entry in session.entries] == ["Ana", "Bob"]
