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

from biaas.config import CATALOG_LIST_URL
from biaas.exceptions import ExternalAPIError
from biaas.faiss_index import FAISSIndex


class APIQueryAgent:
    SIMILARITY_THRESHOLD = 0.45

    def __init__(self, faiss_index: FAISSIndex, sentence_model: SentenceTransformer) -> None:
        self.model = sentence_model
        self.faiss_index = faiss_index

    def get_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, normalize_embeddings=True)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        return embedding.astype(np.float32)

    def search_dataset(self, query: str, top_k: int = 3) -> list[dict[str, Any]] | None:
        if not self.faiss_index.is_ready():
            return None

        query_embedding = self.get_embedding(query)

        try:
            return self.faiss_index.search(query_embedding, top_k=top_k)
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
                response = client.get(endpoint)
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
