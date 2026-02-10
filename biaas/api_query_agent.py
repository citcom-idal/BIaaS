import logging
from typing import Any

import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from biaas.config import BASE_URL
from biaas.faiss_index import FAISSIndex


class APIQueryAgent:
    SIMILARITY_THRESHOLD = 0.45

    def __init__(self, faiss_index: FAISSIndex, sentence_model: SentenceTransformer) -> None:
        self.model = sentence_model
        self.faiss_index = faiss_index

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True).numpy()

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
        wait=wait_exponential(min=1, max=6),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def _fetch_api_data(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        try:
            response = requests.get(endpoint, params=params, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API Fetch Error: {e}")
            st.error(f"API Error: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
    def export_dataset(self, dataset_id: str) -> bytes | None:
        endpoint = f"{BASE_URL}catalog/datasets/{dataset_id}/exports/csv"
        params = {"delimiter": ";"}
        try:
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
            if "text/csv" in response.headers.get("Content-Type", ""):
                return response.content
            else:
                params = {}
                response = requests.get(endpoint, params=params, timeout=60)
                response.raise_for_status()
                if "text/csv" in response.headers.get("Content-Type", ""):
                    return response.content
                else:
                    return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error descargando {dataset_id}: {e}")
            raise
