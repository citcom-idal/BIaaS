from typing import Any

import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from app.core.config import DATASET_SIMILARITY_THRESHOLD
from app.core.exceptions import DatasetNotFoundError, LLMModelError
from app.llm.factory import get_llm_model
from app.schemas.dataset import DatasetMetadata, DatasetSearchResult
from app.services.faiss_service import FaissService


class DatasetService:
    def __init__(
        self, sentence_transformer: SentenceTransformer, faiss_service: FaissService
    ) -> None:
        self.sentence_transformer = sentence_transformer
        self.faiss_service = faiss_service
        self.llm_model = get_llm_model()

    def search_dataset(self, query: str, top_k: int = 3) -> list[DatasetSearchResult] | None:
        query_embedding = self.sentence_transformer.encode(query, normalize_embeddings=True)

        try:
            return self.faiss_service.search(query_embedding, top_k=top_k)
        except Exception as e:
            raise DatasetNotFoundError(f"Error FAISS search: {e}")

    def validate_relevance(self, query: str, dataset_search_result: DatasetSearchResult) -> bool:
        if dataset_search_result.similarity < DATASET_SIMILARITY_THRESHOLD:
            return False

        prompt = f"""Evalúa la relevancia. Consulta: "{query}". Dataset: Título="{dataset_search_result.metadata.title}", Desc="{dataset_search_result.metadata.description[:300]}". ¿Es este dataset ALTAMENTE relevante para la consulta? Responde solo con 'Sí' o 'No'."""

        try:
            raw_response = self.llm_model.get_raw_response(prompt)
        except LLMModelError:
            return False

        return "sí" in raw_response.lower() or "si" in raw_response.lower()

    def __process_dataset(self, dataset_info: dict[str, Any]):
        dataset = dataset_info.get("dataset", {})
        dataset_id = dataset.get("dataset_id", "")
        meta = dataset.get("metas", {}).get("default", {})
        title = meta.get("title", "Sin título")
        description_html = meta.get("description", "")
        description = (
            BeautifulSoup(description_html, "html.parser").get_text().strip()
            if description_html
            else ""
        )

        return DatasetMetadata(id=dataset_id, title=title, description=description)

    def generate_dataset_embeddings(self, datasets: Any):
        texts_for_page: list[str] = []
        embeddings: list[np.ndarray] = []
        embeddings_metadata: list[DatasetMetadata] = []

        for dataset_info in datasets:
            dataset = self.__process_dataset(dataset_info)

            texts_for_page.append(dataset.header())
            embeddings_metadata.append(dataset)

        if texts_for_page:
            page_embeddings = self.sentence_transformer.encode(
                texts_for_page, normalize_embeddings=True, show_progress_bar=False
            )
            embeddings.extend(page_embeddings)

        return embeddings, embeddings_metadata
