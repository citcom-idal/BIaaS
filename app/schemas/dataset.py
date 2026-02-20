from dataclasses import dataclass

import faiss
from pydantic import BaseModel


class DatasetMetadata(BaseModel):
    id: str
    title: str
    description: str


class DatasetSearchResult(BaseModel):
    metadata: DatasetMetadata
    similarity: float


@dataclass(frozen=True)
class DatasetIndexState:
    index: faiss.Index
    metadata: list[DatasetMetadata]
    mtime_index: float
    mtime_metadata: float
