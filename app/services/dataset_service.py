from typing import Any

import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from app.core.exceptions import LLMModelError
from app.llm.factory import get_llm_model
from app.schemas.dataset import DatasetMetadata


class DatasetService:
    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model
        self.llm_model = get_llm_model()

    def validate_relevance(self, query: str, dataset_title: str, dataset_description: str) -> bool:
        prompt = f"""Evalúa la relevancia. Consulta: "{query}". Dataset: Título="{dataset_title}", Desc="{dataset_description[:300]}". ¿Es este dataset ALTAMENTE relevante para la consulta? Responde solo con 'Sí' o 'No'."""

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
            page_embeddings = self.model.encode(
                texts_for_page, normalize_embeddings=True, show_progress_bar=False
            )
            embeddings.extend(page_embeddings)

        return embeddings, embeddings_metadata
