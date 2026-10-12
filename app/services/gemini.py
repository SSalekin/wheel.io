"""The app's single point of contact with the Gemini API.

Every Gemini call in the app goes through the three public functions here;
nothing else may talk to the API directly. All failures — missing config,
timeouts, non-200s, unparseable or invalid replies — surface as
``GeminiError`` so callers never see a raw httpx or pydantic exception.
"""

import base64
import json
import os
from typing import get_args

import httpx
from pydantic import ValidationError

from app.models import Intent, Table
from app.services.mime import detect_mime

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# gemini-2.5-flash is retired for new API keys (HTTP 404: "no longer
# available to new users"). gemini-3.8-flash is the API's suggested
# successor but flaked live (read-timeout, 503) at decision time;
# 3.5-flash-lite answered schema-constrained calls in ~1s, so it ships.
_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_DEFAULT_TIMEOUT_S = 10.0
# State-summary labels beyond this are dropped from the prompt.
_MAX_STATE_ENTRIES = 50

# Wire-level responseSchema per call shape. responseMimeType alone only
# promises *some* JSON — a chatty reply like [{"response": "hi"}] parses
# and then fails model validation, surfacing as "AI unreachable".
# The action enum is single-sourced from the Intent model's Literal.
_INTENT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "action": {
            "type": "STRING",
            "enum": list(get_args(Intent.model_fields["action"].annotation)),
        },
        "entries": {"type": "ARRAY", "items": {"type": "STRING"}, "nullable": True},
        "labels": {"type": "ARRAY", "items": {"type": "STRING"}, "nullable": True},
        "column": {"type": "STRING", "nullable": True},
        "question": {"type": "STRING", "nullable": True},
    },
    "required": ["action"],
}
_STRING_RESPONSE_SCHEMA = {"type": "STRING"}
_STRING_LIST_RESPONSE_SCHEMA = {"type": "ARRAY", "items": {"type": "STRING"}}


class GeminiError(Exception):
    """A Gemini call failed: missing key, transport error, or bad reply."""


def ask_intent(message: str, state_summary: dict) -> Intent:
    """Classify a chat message into an ``Intent``.

    ``state_summary`` carries ``entries`` (labels, capped at 50 here),
    ``picked_count``, ``has_table``, ``table_headers`` and ``remove_picked``.
    An empty ``entries`` list still validates — rejecting it is the
    router's (Task 7) job, not this client's.
    """
    summary = dict(state_summary)
    entries = summary.get("entries")
    if isinstance(entries, list):
        summary["entries"] = entries[:_MAX_STATE_ENTRIES]
    prompt = (
        "Classify the user's message about a random-name wheel app.\n"
        f"User message:\n{message}\n"
        f"Current wheel state as JSON:\n{json.dumps(summary, ensure_ascii=False)}\n"
        "Respond with ONLY a JSON object matching this JSON schema — no "
        "prose, no markdown fences:\n"
        f"{json.dumps(Intent.model_json_schema())}"
    )
    raw = _generate([{"text": prompt}], _INTENT_RESPONSE_SCHEMA)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"Gemini intent reply is not JSON: {raw!r}") from exc
    try:
        return Intent.model_validate(data)
    except ValidationError as exc:
        raise GeminiError(f"Gemini intent reply failed validation: {exc}") from exc


def ask_column(table: Table, message: str) -> str:
    """Pick which of ``table.headers`` the user wants to switch to."""
    prompt = (
        "The user wants to switch the wheel to a different table column.\n"
        f"User message:\n{message}\n"
        f"Available column headers as JSON: {json.dumps(table.headers)}\n"
        'Respond with ONLY a JSON string containing exactly one of those '
        'headers, e.g. "Name" — no prose, no markdown fences.'
    )
    raw = _generate([{"text": prompt}], _STRING_RESPONSE_SCHEMA)
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"Gemini column reply is not JSON: {raw!r}") from exc
    if not isinstance(header, str) or header not in table.headers:
        raise GeminiError(f"Gemini chose unknown column {header!r}.")
    return header


def ask_vision(image: bytes, message: str) -> list[str]:
    """Extract labels from an image; always returns a non-empty ``list[str]``."""
    image_part = {
        "inline_data": {
            "mime_type": detect_mime(image),
            "data": base64.b64encode(image).decode("ascii"),
        }
    }
    prompt_part = {
        "text": (
            "Extract the labels in this image for a random-name wheel.\n"
            f"User message:\n{message}\n"
            'Respond with ONLY a JSON array of non-empty strings, one label '
            'per item, e.g. ["Ana","Bob"] — no prose, no markdown fences.'
        )
    }
    raw = _generate([image_part, prompt_part], _STRING_LIST_RESPONSE_SCHEMA)
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"Gemini vision reply is not JSON: {raw!r}") from exc
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label for label in labels)
    ):
        raise GeminiError(f"Gemini vision reply is not a list of labels: {labels!r}")
    return labels


def _generate(parts: list[dict], response_schema: dict) -> str:
    """POST one schema-constrained ``generateContent`` request; return the candidate text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not set.")
    try:
        timeout = float(os.environ.get("GEMINI_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
    except ValueError as exc:
        raise GeminiError(f"GEMINI_TIMEOUT_S is not a number: {exc}") from exc
    model = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }
    try:
        response = httpx.post(
            f"{_API_BASE}/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            json=payload,
            timeout=timeout,
        )
    except httpx.TimeoutException as exc:
        raise GeminiError(f"Gemini request timed out after {timeout}s.") from exc
    except httpx.HTTPError as exc:
        raise GeminiError(f"Gemini request failed: {exc}") from exc
    if response.status_code != 200:
        raise GeminiError(f"Gemini returned HTTP {response.status_code}.")
    try:
        body = response.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise GeminiError("Gemini response carries no candidate text.") from exc
    if not isinstance(text, str):
        raise GeminiError("Gemini response carries no candidate text.")
    return text
