from pydantic import BaseModel


class Entry(BaseModel):
    id: str
    label: str
    image_ref: str | None = None
