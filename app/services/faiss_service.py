import json
import logging

import faiss
import numpy as np
from pydantic import TypeAdapter

from app.core.config import INDEX_FILE, METADATA_FILE
from app.schemas.dataset import DatasetMetadata, DatasetSearchResult


class FaissService:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.metadata: list[DatasetMetadata] = []

    def load_index(self) -> None:
        try:
            self.index = faiss.read_index(str(INDEX_FILE))

            with open(METADATA_FILE, encoding="utf-8") as f:
                metata_json = json.load(f)
                adapter = TypeAdapter(list[DatasetMetadata])
                self.metadata = adapter.validate_python(metata_json)

            logging.info(
                f"Índice FAISS y metadatos JSON cargados correctamente. {self.index.ntotal} vectores."
            )
        except FileNotFoundError:
            logging.warning(
                f"No se encontraron los ficheros del índice en {INDEX_FILE} o {METADATA_FILE}"
            )
            self.index = None
            self.metadata = []
        except Exception as e:
            logging.error(f"Error al cargar índice o metadatos: {e}")
            self.index = None
            self.metadata = []

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> list[DatasetSearchResult]:
        if not self.is_ready():
            return []

        norm = np.linalg.norm(query_embedding)

        if norm == 0:
            return []

        query_embedding_norm = (query_embedding / norm).astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding_norm, top_k)

        results: list[DatasetSearchResult] = []

        if indices.size > 0:
            for i, idx_val in enumerate(indices[0]):
                if idx_val != -1 and idx_val < len(self.metadata):
                    similarity_score = 1 - (distances[0][i] ** 2) / 2
                    results.append(
                        DatasetSearchResult(
                            metadata=self.metadata[idx_val], similarity=float(similarity_score)
                        )
                    )

        return results

    def process_and_save_index(
        self, dataset_embeddings: list[np.ndarray], dataset_metadatas: list[DatasetMetadata]
    ) -> int:
        embeddings_np = np.array(dataset_embeddings).astype("float32")
        d = embeddings_np.shape[1]

        index = faiss.IndexFlatL2(d)
        index.add(embeddings_np)

        tmp_index_file = INDEX_FILE.with_suffix(".tmp")
        tmp_metadata_file = METADATA_FILE.with_suffix(".tmp")

        faiss.write_index(index, str(tmp_index_file))

        with open(tmp_metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                [metadata.model_dump() for metadata in dataset_metadatas],
                f,
                ensure_ascii=False,
                indent=2,
            )

        tmp_index_file.replace(INDEX_FILE)
        tmp_metadata_file.replace(METADATA_FILE)

        return index.ntotal
