import io
from typing import Any

import httpx
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import CATALOG_LIST_URL
from app.core.exceptions import ExternalAPIError
from app.services.faiss_service import FaissService


class APIQueryAgent:
    SIMILARITY_THRESHOLD = 0.45

    def __init__(self, faiss_service: FaissService, sentence_model: SentenceTransformer) -> None:
        self.model = sentence_model
        self.faiss_service = faiss_service

    def get_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, normalize_embeddings=True)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        return embedding.astype(np.float32)

    def search_dataset(self, query: str, top_k: int = 3) -> list[dict[str, Any]] | None:
        if not self.faiss_service.is_ready():
            return None

        query_embedding = self.get_embedding(query)

        try:
            return self.faiss_service.search(query_embedding, top_k=top_k)
        except Exception as e:
            st.error(f"APIQueryAgent: Error FAISS search: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(
            [httpx.RequestError, httpx.HTTPStatusError, httpx.TimeoutException, Exception]
        ),
    )
    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        endpoint = f"{CATALOG_LIST_URL}/{dataset_id}/exports/csv"

        with httpx.Client(timeout=60) as client:
            try:
                response = client.get(endpoint, params={"delimiter": ";"})
                response.raise_for_status()
            except httpx.TimeoutException as e:
                raise ExternalAPIError(f"Timeout al descargar {dataset_id}: {e}")
            except httpx.HTTPStatusError as e:
                raise ExternalAPIError(
                    f"Error HTTP {e.response.status_code} al descargar {dataset_id}: {e}"
                )
            except httpx.RequestError as e:
                raise ExternalAPIError(f"Error de conexión al descargar {dataset_id}: {e}")
            except Exception as e:
                raise ExternalAPIError(f"Error inesperado al descargar {dataset_id}: {e}")

        df = pd.read_csv(io.StringIO(response.text), delimiter=";")

        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

        return df

    def fetch_datasets_count(self) -> int:
        with httpx.Client(timeout=20) as client:
            try:
                params = {"limit": 1, "offset": 0}
                response = client.get(CATALOG_LIST_URL, params=params)
                response.raise_for_status()
            except httpx.RequestError as e:
                raise ExternalAPIError(f"Error de conexión al obtener lista de datasets: {e}")
            except httpx.HTTPStatusError as e:
                raise ExternalAPIError(
                    f"Error HTTP {e.response.status_code} al obtener lista de datasets: {e}"
                )
            except httpx.TimeoutException as e:
                raise ExternalAPIError(f"Timeout al obtener lista de datasets: {e}")
            except Exception as e:
                raise ExternalAPIError(f"Error inesperado al obtener lista de datasets: {e}")

        data = response.json()

        return data.get("total_count", 0)

    def fetch_datasets_page(self, limit: int = 100, offset: int = 0) -> httpx.Response:
        with httpx.Client(timeout=30) as client:
            try:
                params = {"limit": limit, "offset": offset}
                response = client.get(CATALOG_LIST_URL, params=params)
                response.raise_for_status()
            except httpx.RequestError as e:
                raise ExternalAPIError(f"Error de conexión al obtener lista de datasets: {e}")
            except httpx.HTTPStatusError as e:
                raise ExternalAPIError(
                    f"Error HTTP {e.response.status_code} al obtener lista de datasets: {e}"
                )
            except httpx.TimeoutException as e:
                raise ExternalAPIError(f"Timeout al obtener lista de datasets: {e}")
            except Exception as e:
                raise ExternalAPIError(f"Error inesperado al obtener lista de datasets: {e}")

        return response
