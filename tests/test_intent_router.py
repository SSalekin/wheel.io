"""Tests for the intent router: rules fast path, Gemini escalation, degradation."""

import json

import httpx
import pytest
import respx

from app.models import Entry, Table, TableColumn, WheelSession
from app.services.intent_router import _degrade, interpret

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-3.5-flash-lite:generateContent"
)

WARNING = "⚠️ AI unreachable — best-guess parsing."
CLARIFY_DEFAULT = "Could you send me a list or an image to put on the wheel?"


def _session(**kwargs) -> WheelSession:
    return WheelSession(token="t", created_at=0.0, **kwargs)


def _candidate_body(text: str) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


@pytest.fixture(autouse=True)
def gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT_S", raising=False)


@respx.mock
def test_rules_fast_path_no_http():
    respx.post(API_URL)

    intent, warnings = interpret("wheel:\nAna\nBob", 0, _session())

    assert intent.action == "create_text"
    assert intent.entries == ["Ana", "Bob"]
    assert warnings == []
    # Brief's "respx assert_no_call": the fast path never talks to Gemini.
    respx.calls.assert_not_called()


@respx.mock
def test_gemini_called_when_rules_miss():
    respx.post(API_URL).respond(
        200, json=_candidate_body('{"action":"remove","labels":["Bob"]}')
    )
    session = _session(
        entries=[Entry(id="1", label="Ana"), Entry(id="2", label="Bob")],
        source_table=Table(
            columns=[TableColumn(header="Name", values=["Ana", "Bob"])],
            confidence=0.9,
        ),
        remove_picked=True,
    )

    intent, warnings = interpret("hey, take Bob off the wheel", 0, session)

    assert intent.action == "remove"
    assert intent.labels == ["Bob"]
    assert warnings == []
    # state_summary contract: the five keys Task 6's client expects.
    payload = json.loads(respx.calls.last.request.content)
    prompt = payload["contents"][0]["parts"][0]["text"]
    assert '"entries": ["Ana", "Bob"]' in prompt
    assert '"picked_count": 0' in prompt
    assert '"has_table": true' in prompt
    assert '"table_headers": ["Name"]' in prompt
    assert '"remove_picked": true' in prompt


@respx.mock
def test_gemini_timeout_clarifies():
    respx.post(API_URL).side_effect = httpx.TimeoutException("timed out")

    intent, warnings = interpret("please pick students for me", 0, _session())

    assert intent.action == "clarify"
    assert intent.question == CLARIFY_DEFAULT
    assert warnings == [WARNING]


@respx.mock
def test_gemini_timeout_with_list_degrades_to_create_text():
    # "Ana, Bob" is a comma-only list without a verb, so rules return None.
    respx.post(API_URL).side_effect = httpx.TimeoutException("timed out")

    intent, warnings = interpret("Ana, Bob", 0, _session())

    assert intent.action == "create_text"
    assert intent.entries == ["Ana", "Bob"]
    assert warnings == [WARNING]


@respx.mock
def test_gemini_timeout_prose_clarifies_not_junk_wheel():
    # Regression: prose with an instruction word in it ("hey, take Ana off
    # the wheel") used to degrade into a junk wheel with two entries; it
    # clarifies instead of guessing.
    respx.post(API_URL).side_effect = httpx.TimeoutException("timed out")

    intent, warnings = interpret("hey, take Ana off the wheel", 0, _session())

    assert intent.action == "clarify"
    assert intent.question == CLARIFY_DEFAULT
    assert warnings == [WARNING]


@respx.mock
def test_gemini_unknown_action_degrades():
    respx.post(API_URL).respond(200, json=_candidate_body('{"action":"dance"}'))

    intent, warnings = interpret("just do something fun", 0, _session())

    assert intent.action == "clarify"
    assert intent.question == CLARIFY_DEFAULT
    assert warnings == [WARNING]


@respx.mock
def test_gemini_empty_entries_degrades():
    respx.post(API_URL).respond(
        200, json=_candidate_body('{"action":"create_text","entries":[]}')
    )

    intent, warnings = interpret("make a wheel", 0, _session())

    assert intent.action == "clarify"
    assert intent.question == CLARIFY_DEFAULT
    assert warnings == [WARNING]


def test_degrade_single_image_is_create_image_with_warning():
    # The image fast-path normally short-circuits before Gemini, so the
    # degrade branches are pinned directly: unreachable AI plus one
    # attached image still yields create_image + the exact warning.
    intent, warnings = _degrade("please pick students for me", 1)

    assert intent.action == "create_image"
    assert warnings == [WARNING]


def test_degrade_two_images_is_create_images_with_warning():
    intent, warnings = _degrade("please pick students for me", 2)

    assert intent.action == "create_images"
    assert warnings == [WARNING]


def test_bare_column_name_switches_locally_without_ai():
    # A follow-up naming exactly one column of the session's table needs
    # no AI: the OCR already read the headers, so the router switches
    # locally instead of escalating (reported: "AI unreachable" when the
    # user typed the column name after uploading).
    session = _session()
    session.source_table = Table(
        columns=[
            TableColumn(header="Lecturer", values=["Dr Smith"]),
            TableColumn(header="Course", values=["Maths"]),
        ],
        confidence=95.0,
    )

    intent, warnings = interpret("Lecturer", 0, session)

    assert intent.action == "switch_column"
    assert intent.column == "Lecturer"
    assert warnings == []


def test_punctuated_column_name_still_switches_locally():
    # OCR and sloppy typing add punctuation: "lecturer." must still name
    # the Lecturer column.
    session = _session()
    session.source_table = Table(
        columns=[
            TableColumn(header="Lecturer", values=["Dr Smith"]),
            TableColumn(header="Course", values=["Maths"]),
        ],
        confidence=95.0,
    )

    intent, warnings = interpret("Lecturer.", 0, session)

    assert intent.action == "switch_column"
    assert intent.column == "Lecturer"
    assert warnings == []


@respx.mock
def test_two_named_columns_stay_ambiguous():
    # Naming two columns is genuinely ambiguous: never guess locally —
    # fall through to the normal flow (Gemini online, degrade offline).
    respx.post(API_URL).side_effect = httpx.TimeoutException("timed out")
    session = _session()
    session.source_table = Table(
        columns=[
            TableColumn(header="Lecturer", values=["Dr Smith"]),
            TableColumn(header="Course", values=["Maths"]),
        ],
        confidence=95.0,
    )

    intent, warnings = interpret("Course or Lecturer", 0, session)

    assert intent.action == "clarify"
    assert warnings == [WARNING]
