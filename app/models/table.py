from pydantic import BaseModel


class TableColumn(BaseModel):
    header: str
    values: list[str]


class Table(BaseModel):
    columns: list[TableColumn]
    confidence: float

    @property
    def headers(self) -> list[str]:
        return [column.header for column in self.columns]

    def get_column(self, header: str) -> TableColumn | None:
        for column in self.columns:
            if column.header == header:
                return column
        return None
