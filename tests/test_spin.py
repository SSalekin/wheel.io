import re

from app.main import app

SESSION_EXPIRED = "Session expired — refresh the page to start over."
EMPTY_WHEEL = "The wheel is empty — add some entries first."


def _page(client) -> tuple[str, str]:
    """Load GET / and pull the per-page hidden token plus the rendered body."""
    body = client.get("/").text
    match = re.search(r'name="token" value="([^"]+)"', body)
    assert match, "page must render a hidden token input with a non-empty value"
    return match.group(1), body


def _token_with_entries(client) -> str:
    token, _page_body = _page(client)
    response = client.post(
        "/chat",
        data={"token": token, "message": "wheel:\nAna\nBob\nChi"},
    )
    assert response.status_code == 200
    return token


def test_spin_endpoint_updates_banner_and_wheel(client):
    token = _token_with_entries(client)

    response = client.post("/spin", data={"token": token})

    assert response.status_code == 200
    body = response.text
    assert "🎉" in body
    assert 'data-animate="1"' in body
    assert re.search(r'data-pre-index="\d+"', body)
    # Go button wiring lives in the go-form partial (ruling: partials are the
    # single source of truth), which the spin response swaps back OOB.
    assert 'hx-post="/spin"' in body
    assert 'hx-target="#banner"' in body


def test_spin_empty_endpoint_bubbles(client):
    token, _body = _page(client)

    response = client.post("/spin", data={"token": token})

    assert response.status_code == 200
    assert EMPTY_WHEEL in response.text


def test_stale_token_on_spin(client):
    response = client.post("/spin", data={"token": "nope"})

    assert response.status_code == 200
    assert SESSION_EXPIRED in response.text
    assert app.state.store.get("nope") is None


def test_toggle_flips_state(client):
    token, body = _page(client)
    assert app.state.store.get(token).remove_picked is False
    # The checkbox's hx wiring lives on the go-form region of the page.
    assert 'hx-post="/toggle"' in body
    assert 'hx-swap="none"' in body

    response = client.post("/toggle", data={"token": token, "remove_picked": "true"})

    assert response.status_code == 200
    assert response.text == ""
    assert app.state.store.get(token).remove_picked is True


def test_go_disabled_when_empty(client):
    token, _body = _page(client)

    response = client.post("/spin", data={"token": token})

    assert response.status_code == 200
    assert re.search(r'<button[^>]*class="go-button"[^>]*\bdisabled\b', response.text)
