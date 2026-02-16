import json
import logging
import os
from typing import Any

import faiss
import numpy as np

from app.core.config import INDEX_FILE, METADATA_FILE
from app.schemas.dataset import DatasetMetadata


class FaissService:
    def __init__(self, index_path: str = INDEX_FILE, metadata_path: str = METADATA_FILE):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata: list[Any] = []

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

    def process_and_save_index(
        self, dataset_embeddings: list[np.ndarray], dataset_metadatas: list[DatasetMetadata]
    ) -> int:
        embeddings_np = np.array(dataset_embeddings).astype("float32")
        d = embeddings_np.shape[1]

        index = faiss.IndexFlatL2(d)
        index.add(embeddings_np)

        faiss.write_index(index, INDEX_FILE)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [metadata.model_dump() for metadata in dataset_metadatas],
                f,
                ensure_ascii=False,
                indent=2,
            )

        return index.ntotal
