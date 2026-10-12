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
    # Winner label parked at spin time: the banner is served by GET /banner
    # when the animation ends, so the announcement never rides the spin
    # response (it would land in #banner mid-spin and give the answer away).
    pending_banner: str | None = None
