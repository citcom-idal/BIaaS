import json
import logging
import os
from typing import Any

import faiss
import numpy as np

from biaas.config import INDEX_FILE, METADATA_FILE


class FAISSIndex:
    def __init__(self, index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata: list[Any] = []
        self.load_index()

    def load_index(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logging.info(
                    f"Índice FAISS y metadatos JSON cargados correctamente. {self.index.ntotal} vectores."
                )
            except Exception as e:
                logging.error(f"Error al cargar índice o metadatos: {e}")
                self.index = None
                self.metadata = []
        else:
            logging.warning(
                f"No se encontraron los ficheros del índice en {self.index_path} o {self.metadata_path}"
            )
            self.index = None

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.is_ready():
            return []
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            return []
        query_embedding_norm = (query_embedding / norm).astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding_norm, top_k)

        results = []
        if indices.size > 0:
            for i, idx_val in enumerate(indices[0]):
                if idx_val != -1 and idx_val < len(self.metadata):
                    similarity_score = 1 - (distances[0][i] ** 2) / 2
                    results.append(
                        {"metadata": self.metadata[idx_val], "similarity": float(similarity_score)}
                    )
        return results
