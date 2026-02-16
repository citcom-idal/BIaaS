from pydantic import BaseModel


class DatasetMetadata(BaseModel):
    id: str
    title: str
    description: str

    def header(self) -> str:
        return f"título: {self.title}; descripción: {self.description}"


class DatasetSearchResult(BaseModel):
    metadata: DatasetMetadata
    similarity: float
