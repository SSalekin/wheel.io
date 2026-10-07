from typing import Literal

from pydantic import BaseModel


class Intent(BaseModel):
    action: Literal[
        "create_text",
        "create_images",
        "create_image",
        "add",
        "remove",
        "switch_column",
        "reset",
        "spin",
        "clarify",
    ]
    entries: list[str] | None = None
    labels: list[str] | None = None
    column: str | None = None
    question: str | None = None
