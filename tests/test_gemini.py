"""Tests for the Gemini client — all HTTP mocked with respx."""

import base64
import json

import httpx
import pytest
import respx

from app.models import Intent, Table, TableColumn
from app.services.gemini import GeminiError, ask_column, ask_intent, ask_vision

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-2.5-flash:generateContent"
)

STATE_SUMMARY = {
    "entries": ["Ana", "Bob"],
    "picked_count": 0,
    "has_table": False,
    "table_headers": [],
    "remove_picked": False,
}


def _candidate_body(text: str) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


def _table() -> Table:
    return Table(
        columns=[
            TableColumn(header="Name", values=["Ana", "Bob"]),
            TableColumn(header="ID", values=["7", "8"]),
        ],
        confidence=0.9,
    )


@pytest.fixture(autouse=True)
def gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT_S", raising=False)


@respx.mock
def test_ask_intent_success():
    respx.post(API_URL).respond(
        200, json=_candidate_body('{"action":"clarify","question":"Which column?"}')
    )

    intent = ask_intent("clarify: which column?", dict(STATE_SUMMARY))

    assert isinstance(intent, Intent)
    assert intent.action == "clarify"
    assert intent.question == "Which column?"

    payload = json.loads(respx.calls.last.request.content)
    prompt = payload["contents"][0]["parts"][0]["text"]
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    # The prompt carries the message, the state summary, the Intent
    # schema, and an instruction to answer in JSON only.
    assert "clarify: which column?" in prompt
    assert '"picked_count": 0' in prompt
    assert '"action"' in prompt
    assert "JSON" in prompt


@respx.mock
def test_ask_intent_unknown_action_raises():
    respx.post(API_URL).respond(200, json=_candidate_body('{"action":"explode"}'))

    with pytest.raises(GeminiError):
        ask_intent("explode everything", dict(STATE_SUMMARY))


@respx.mock
def test_ask_intent_returns_model_for_empty_entries():
    """Empty entries validate as an Intent; rejecting them is Task 7's job."""
    respx.post(API_URL).respond(
        200, json=_candidate_body('{"action":"create_text","entries":[]}')
    )

    intent = ask_intent("make a wheel", dict(STATE_SUMMARY))

    assert isinstance(intent, Intent)
    assert intent.action == "create_text"
    assert intent.entries == []


@respx.mock
def test_ask_column_validates_header():
    route = respx.post(API_URL)
    route.side_effect = [
        httpx.Response(200, json=_candidate_body('"Student ID"')),
        httpx.Response(200, json=_candidate_body('"ID"')),
    ]
    table = _table()

    with pytest.raises(GeminiError):
        ask_column(table, "switch to the student id one")

    assert ask_column(table, "switch to id") == "ID"


@respx.mock
def test_ask_vision_empty_list_raises():
    respx.post(API_URL).respond(200, json=_candidate_body("[]"))

    with pytest.raises(GeminiError):
        ask_vision(b"fake-png-bytes", "how many names are in this image?")


@respx.mock
def test_ask_vision_success():
    respx.post(API_URL).respond(200, json=_candidate_body('["Ana", "Bob"]'))

    labels = ask_vision(b"fake-png-bytes", "names from this image")

    assert labels == ["Ana", "Bob"]

    payload = json.loads(respx.calls.last.request.content)
    parts = payload["contents"][0]["parts"]
    image_part = next(part for part in parts if "inline_data" in part)
    assert image_part["inline_data"]["mime_type"] == "image/png"
    assert image_part["inline_data"]["data"] == base64.b64encode(
        b"fake-png-bytes"
    ).decode("ascii")


@respx.mock
def test_ask_vision_detects_mime_from_bytes():
    respx.post(API_URL).respond(200, json=_candidate_body('["Ana"]'))

    cases = [
        (b"\xff\xd8\xff\xe0fake-jpeg", "image/jpeg"),
        (b"RIFF\x10\x00\x00\x00WEBPfake", "image/webp"),
        (b"\x89PNG\r\n\x1a\nfake-png", "image/png"),
        (b"mystery-bytes", "image/png"),  # unknown ⇒ default
    ]
    for image, expected_mime in cases:
        ask_vision(image, "labels from this image")
        payload = json.loads(respx.calls.last.request.content)
        parts = payload["contents"][0]["parts"]
        image_part = next(part for part in parts if "inline_data" in part)
        assert image_part["inline_data"]["mime_type"] == expected_mime, expected_mime


@respx.mock
def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    respx.post(API_URL).respond(200, json=_candidate_body('{"action":"spin"}'))

    with pytest.raises(GeminiError):
        ask_intent("spin", dict(STATE_SUMMARY))
    with pytest.raises(GeminiError):
        ask_column(_table(), "switch to id")
    with pytest.raises(GeminiError):
        ask_vision(b"fake-png-bytes", "names from this image")

    assert not respx.calls


@respx.mock
def test_timeout_raises():
    respx.post(API_URL).side_effect = httpx.TimeoutException("timed out")

    with pytest.raises(GeminiError):
        ask_intent("spin", dict(STATE_SUMMARY))


# Contract coverage for the Interfaces line: every path — non-200 and
# unparseable body included — must surface as GeminiError.


@respx.mock
def test_non_200_raises():
    respx.post(API_URL).respond(500, json={"error": {"message": "boom"}})

    with pytest.raises(GeminiError):
        ask_intent("spin", dict(STATE_SUMMARY))


@respx.mock
def test_unparseable_body_raises():
    respx.post(API_URL).respond(200, json={"unexpected": "shape"})

    with pytest.raises(GeminiError):
        ask_intent("spin", dict(STATE_SUMMARY))


@respx.mock
def test_ask_intent_honors_model_override(monkeypatch):
    # GEMINI_MODEL selects the wire target; the default model must not
    # shadow the override.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/gemini-custom-model:generateContent"
    ).respond(200, json=_candidate_body('{"action":"clarify"}'))

    ask_intent("hello", dict(STATE_SUMMARY))

    assert route.called
