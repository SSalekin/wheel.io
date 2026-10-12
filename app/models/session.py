from pydantic import BaseModel

from app.models.entry import Entry
from app.models.table import Table


class WheelSession(BaseModel):
    token: str
    created_at: float
    entries: list[Entry] = []
    picked: list[Entry] = []
    remove_picked: bool = False
    source_table: Table | None = None
    media: dict[str, bytes] = {}
