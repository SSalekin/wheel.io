"""Choose which table column becomes the wheel's entries.

Decision order — the first rule that applies wins:

1. A message word equal to a header (case-insensitive) names the column.
2. A keyword mention (``email`` / ``id`` / ``name|student``) selects the
   column whose header carries a token from the matching header set.
3. A lone column needs no guessing.
4. Anything ambiguous is asked of Gemini.
5. Gemini unreachable → the first column with non-empty values, plus the
   exact warning.

Columns whose values are all empty are never chosen by steps 2–5.
"""

from typing import Literal

from pydantic import BaseModel

from app.models import Table, TableColumn
from app.services import gemini
from app.services.gemini import GeminiError

EMAIL_HEADERS = {"email", "e-mail", "mail"}
ID_HEADERS = {"id", "roll", "enrollment", "admission"}
NAME_HEADERS = {"name", "student", "full"}

_AI_UNREACHABLE_WARNING = "⚠️ AI unreachable — best-guess parsing."

# (message mentions, header tokens) in the brief's order: email → id → name.
_KEYWORD_GROUPS: tuple[tuple[tuple[str, ...], set[str]], ...] = (
    (("email",), EMAIL_HEADERS),
    (("id",), ID_HEADERS),
    (("name", "student"), NAME_HEADERS),
)


class ColumnChoice(BaseModel):
    """The column picked for the wheel, plus how it was chosen."""

    values: list[str]
    source: Literal["header", "gemini", "first"]
    warning: str | None = None


def pick_column(table: Table, message: str) -> ColumnChoice:
    """Decide which of ``table``'s columns becomes the wheel entries."""
    lowered = message.lower()
    tokens = set(lowered.split())

    # 1. The message names a header outright.
    for column in table.columns:
        if column.header.lower() in tokens:
            return ColumnChoice(values=column.values, source="header")

    # 2. Keyword heuristics: a message token starting with the mention →
    #    header token match. Prefixes, not substrings: "provide" must not
    #    trigger the `id` group, while "ids" still does.
    for mentions, header_tokens in _KEYWORD_GROUPS:
        if not any(
            token.startswith(mention) for token in tokens for mention in mentions
        ):
            continue
        for column in table.columns:
            if _is_empty(column):
                continue  # steps 2–5 never choose an all-empty column
            if set(column.header.lower().split()) & header_tokens:
                return ColumnChoice(values=column.values, source="header")

    # 3. A single non-empty column needs no guessing.
    if len(table.columns) == 1 and not _is_empty(table.columns[0]):
        return ColumnChoice(values=table.columns[0].values, source="first")

    # 4. Ambiguous → Gemini.
    try:
        header = gemini.ask_column(table, message)
    except GeminiError:
        # 5. Unreachable → first column with content + the exact warning.
        column = _first_non_empty(table)
        return ColumnChoice(
            values=column.values, source="first", warning=_AI_UNREACHABLE_WARNING
        )
    # ask_column validated the header against table.headers, so it exists.
    column = table.get_column(header)
    if _is_empty(column):
        # Gemini pointed at an all-empty column — never chosen by steps
        # 2–5 — so fall back like step 5, minus the warning: the call
        # itself succeeded.
        column = _first_non_empty(table)
        return ColumnChoice(values=column.values, source="first")
    return ColumnChoice(values=column.values, source="gemini")


def _is_empty(column: TableColumn) -> bool:
    """A column is empty when every value is blank or whitespace."""
    return not any(value.strip() for value in column.values)


def _first_non_empty(table: Table) -> TableColumn:
    """First column with any content; if every column is empty, the first.

    OCR always yields at least one column, so ``table.columns[0]`` exists.
    """
    for column in table.columns:
        if not _is_empty(column):
            return column
    return table.columns[0]
