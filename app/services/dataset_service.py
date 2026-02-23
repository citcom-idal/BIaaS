import io

import httpx
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.core.config import CATALOG_LIST_URL, DATASET_SIMILARITY_THRESHOLD
from app.core.exceptions import DatasetNotFoundError, ExternalAPIError, LLMModelError
from app.llm import LLMModel
from app.schemas.dataset import DatasetSearchResult
from app.services.faiss_index_service import FaissIndexService
from app.utils import fetch_with_retry


class DatasetService:
    def __init__(
        self,
        sentence_transformer: SentenceTransformer,
        faiss_service: FaissIndexService,
        llm_model: LLMModel,
    ) -> None:
        self.sentence_transformer = sentence_transformer
        self.faiss_service = faiss_service
        self.llm_model = llm_model

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        endpoint = f"{CATALOG_LIST_URL}/{dataset_id}/exports/csv"
        delimiter = ";"
        params = httpx.QueryParams(delimiter=delimiter)

        try:
            response = fetch_with_retry(endpoint, params=params, timeout=httpx.Timeout(60.0))
        except httpx.RequestError as e:
            raise ExternalAPIError(f"API Error: {e}") from e

        df = pd.read_csv(io.BytesIO(response.content), delimiter=delimiter)

        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

        return df

    def search_dataset(self, query: str, top_k: int = 3) -> list[DatasetSearchResult] | None:
        query_embedding = self.sentence_transformer.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )

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
