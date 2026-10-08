"""Tests for the column picker — Gemini HTTP mocked with respx."""

import httpx
import pytest
import respx

from app.models import Table, TableColumn
from app.services.column_picker import ColumnChoice, pick_column

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-2.5-flash:generateContent"
)

# Byte-exact copy of the brief's step-5 warning.
AI_UNREACHABLE_WARNING = "⚠️ AI unreachable — best-guess parsing."


def _candidate_body(text: str) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


def _table(*columns: tuple[str, list[str]]) -> Table:
    return Table(
        columns=[
            TableColumn(header=header, values=values) for header, values in columns
        ],
        confidence=0.9,
    )


@pytest.fixture(autouse=True)
def gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT_S", raising=False)


@respx.mock
def test_header_token_match():
    # Step 1 finds no exact header ("ids" is not a header); step 2's `id`
    # mention matches the "Student ID" header token.
    table = _table(("Name", ["Ana", "Bob"]), ("Student ID", ["7", "8"]))

    choice = pick_column(table, "pick the ids")

    assert isinstance(choice, ColumnChoice)
    assert choice.values == ["7", "8"]
    assert choice.source == "header"
    assert choice.warning is None
    assert not respx.calls  # decided before any Gemini call


@respx.mock
def test_exact_header_in_message():
    # The exact "Email" header must win over the looser "Contact Email"
    # that step 2's keyword containment would have picked instead.
    table = _table(
        ("Contact Email", ["ana@example.com"]),
        ("Email", ["team@example.com"]),
    )

    choice = pick_column(table, "use the Email column")

    assert choice.values == ["team@example.com"]
    assert choice.source == "header"
    assert choice.warning is None
    assert not respx.calls


@respx.mock
def test_single_column_passthrough():
    table = _table(("Cohort", ["Red", "Blue"]))

    choice = pick_column(table, "spin the wheel")

    assert choice.values == ["Red", "Blue"]
    assert choice.source == "first"
    assert choice.warning is None
    assert not respx.calls


@respx.mock
def test_ambiguous_calls_gemini():
    respx.post(API_URL).respond(200, json=_candidate_body('"Section"'))
    table = _table(("Group", ["G1", "G2"]), ("Section", ["S1", "S2"]))

    choice = pick_column(table, "wheel these")

    assert choice.values == ["S1", "S2"]
    assert choice.source == "gemini"
    assert choice.warning is None
    assert len(respx.calls) == 1


@respx.mock
def test_gemini_down_uses_first_text_column_with_warning():
    respx.post(API_URL).side_effect = httpx.TimeoutException("timed out")
    table = _table(("Group", []), ("Section", ["S1", "S2"]))

    choice = pick_column(table, "wheel these")

    assert choice.values == ["S1", "S2"]
    assert choice.source == "first"
    assert choice.warning == AI_UNREACHABLE_WARNING


@respx.mock
def test_empty_column_skipped():
    # Step 2's `name` mention matches both headers ("Name", "Student" —
    # NAME_HEADERS holds "student"), but the all-empty first column is
    # never chosen: the second column wins.
    table = _table(("Name", []), ("Student", ["A1", "A2"]))

    choice = pick_column(table, "wheel the names")

    assert choice.values == ["A1", "A2"]
    assert choice.source == "header"
    assert choice.warning is None
    assert not respx.calls


@respx.mock
def test_email_mention_matches_email_headers():
    table = _table(("Name", ["Ana"]), ("E-mail Address", ["ana@example.com"]))

    choice = pick_column(table, "wheel by email")

    assert choice.values == ["ana@example.com"]
    assert choice.source == "header"
    assert choice.warning is None
    assert not respx.calls


@respx.mock
def test_gemini_pick_skipped_when_column_empty():
    # Gemini names the first column, but its values are all empty — per
    # the contract (steps 2-5) it is never chosen; the first column with
    # content is used instead. The call itself succeeded, so no warning.
    respx.post(API_URL).respond(200, json=_candidate_body('"Group"'))
    table = _table(("Group", []), ("Section", ["S1", "S2"]))

    choice = pick_column(table, "wheel these")

    assert choice.values == ["S1", "S2"]
    assert choice.source == "first"
    assert choice.warning is None
