import json
import re

from app.main import app
from app.models import Entry

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


def _seed(client, labels, remove_picked) -> str:
    """Fresh page token with a deterministic session state."""
    token, _ = _page(client)
    session = app.state.store.get(token)
    assert session is not None
    session.entries = [Entry(id=f"e{i}", label=label) for i, label in enumerate(labels)]
    session.remove_picked = remove_picked
    return token


def _wheel_data_attrs(html: str) -> dict[str, str]:
    node = re.search(r'<div id="wheel-data"([^>]*)>', html)
    assert node, "response must ship the #wheel-data node"
    # Values may be quoted with either delimiter (tojson escapes ' as \u0027,
    # so a single-quoted JSON attribute contains no raw ').
    return {
        name: value
        for name, _quote, value in re.findall(
            r'''(data-[\w-]+)=(['"])(.*?)\2''', node.group(1)
        )
    }


def test_chat_spin_with_removal_ships_pre_removal_snapshot(client):
    # Audit #2: the animated entries, the shipped index, and the glow must
    # all agree; a removed winner cannot be pointed at in the new wheel.
    token = _seed(client, ["Ana"], remove_picked=True)

    response = client.post("/chat", data={"token": token, "message": "spin"})

    attrs = _wheel_data_attrs(response.text)
    # The snapshot the client animates still contains the winner…
    assert "Ana" in attrs["data-spin-entries"]
    assert attrs["data-pre-index"] == "0"
    # …while the shipped (post-removal) wheel is empty…
    assert json.loads(attrs["data-entries"]) == []
    # …and there is no glow: the winner is no longer on the wheel.
    assert "data-glow-index" not in attrs


def test_spin_endpoint_with_removal_ships_pre_removal_snapshot(client):
    token = _seed(client, ["Ana"], remove_picked=True)

    response = client.post("/spin", data={"token": token})

    attrs = _wheel_data_attrs(response.text)
    assert "Ana" in attrs["data-spin-entries"]
    assert attrs["data-pre-index"] == "0"
    assert "data-glow-index" not in attrs


def test_spin_without_removal_glows_the_winner(client):
    token = _seed(client, ["Ana", "Bob"], remove_picked=False)

    response = client.post("/spin", data={"token": token})

    attrs = _wheel_data_attrs(response.text)
    # Nothing removed: the snapshot is the shipped wheel, and the glow is
    # the index the animation lands on.
    assert json.loads(attrs["data-spin-entries"]) == json.loads(attrs["data-entries"])
    assert attrs["data-glow-index"] == attrs["data-pre-index"]


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


def test_spin_defers_picked_update_until_fetched(client):
    # The picked sidebar must not jump while the wheel is still spinning:
    # the spin response omits it, and GET /picked serves the post-spin state.
    token = _token_with_entries(client)
    client.post("/toggle", data={"token": token, "remove_picked": "true"})

    spin_response = client.post("/spin", data={"token": token})

    assert spin_response.status_code == 200
    assert 'id="picked-list"' not in spin_response.text

    picked_response = client.get("/picked", params={"token": token})

    assert picked_response.status_code == 200
    assert 'id="picked-list"' in picked_response.text
    session = app.state.store.get(token)
    assert len(session.picked) == 1
    assert session.picked[0].label in picked_response.text


def test_picked_endpoint_stale_token(client):
    response = client.get("/picked", params={"token": "nope"})

    assert response.status_code == 200
    assert SESSION_EXPIRED in response.text
    assert app.state.store.get("nope") is None


def test_picked_list_is_one_self_contained_element(client):
    # wheel.js parses the /picked response and replaces #picked-list, so the
    # whole sidebar — including the empty state — must live inside the <ol>.
    token, _body = _page(client)

    response = client.get("/picked", params={"token": token})

    assert response.status_code == 200
    match = re.search(
        r'<ol[^>]*id="picked-list"[^>]*>.*?</ol>', response.text, re.S
    )
    assert match, "picked list must render as a single self-contained <ol>"
    assert "picked-empty" in match.group(0)


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
