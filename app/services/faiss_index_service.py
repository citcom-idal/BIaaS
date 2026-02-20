import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError

import faiss
import numpy as np
from pydantic import TypeAdapter

from app.core.config import INDEX_FILE, METADATA_FILE
from app.schemas.dataset import (
    DatasetIndexState,
    DatasetMetadata,
    DatasetSearchResult,
)


class FaissIndexService:
    def __init__(self):
        self.__state: DatasetIndexState | None = None
        self.__lock = threading.RLock()
        self.__load_future: Future | None = None
        self.__executor = ThreadPoolExecutor(max_workers=1)

    def files_exist(self) -> bool:
        return INDEX_FILE.exists() and METADATA_FILE.exists()

    def ensure_index_loaded(self, timeout: float = 30.0) -> None:
        if not self.files_exist():
            return

        with self.__lock:
            need_load = False

            if self.__state is None:
                need_load = True
            else:
                try:
                    index_mtime = os.path.getmtime(INDEX_FILE)
                    metadata_mtime = os.path.getmtime(METADATA_FILE)
                except OSError as e:
                    logging.error(f"Error al obtener la fecha de modificación de los archivos: {e}")
                    need_load = True
                else:
                    need_load = (
                        index_mtime > self.__state.mtime_index
                        or metadata_mtime > self.__state.mtime_metadata
                    )

            if not need_load:
                return

            future = self.__load_future
            if future is None or future.done():
                future = self.__executor.submit(self.__load_index)
                self.__load_future = future

        try:
            future.result(timeout=timeout)
        except TimeoutError:
            logging.warning("Tiempo de espera excedido al cargar el índice FAISS (%.1fs)", timeout)
        except Exception:
            logging.exception("Error al cargar el índice FAISS")
            raise

    def shutdown(self) -> None:
        try:
            self.__executor.shutdown(wait=False)
        except Exception:
            logging.exception("Error al cerrar el ThreadPoolExecutor del FaissIndexService")

    def __load_index(self) -> None:
        start = time.time()

        new_index = faiss.read_index(str(INDEX_FILE))

        with open(METADATA_FILE, encoding="utf-8") as f:
            metata_json = json.load(f)
            adapter = TypeAdapter(list[DatasetMetadata])
            new_metadata = adapter.validate_python(metata_json)

        index_mtime = os.path.getmtime(INDEX_FILE)
        metadata_mtime = os.path.getmtime(METADATA_FILE)

        state = DatasetIndexState(
            index=new_index,
            metadata=new_metadata,
            mtime_index=index_mtime,
            mtime_metadata=metadata_mtime,
        )

        with self.__lock:
            self.__state = state

        elapsed = time.time() - start

        logging.info(
            f"Índice FAISS y metadatos JSON cargados correctamente en {elapsed:.2f}s. {state.index.ntotal} vectores."
        )

    def get_faiss_index_state(self) -> DatasetIndexState:
        with self.__lock:
            if self.__state is None:
                raise RuntimeError("Índice FAISS no cargado.")

            return self.__state

    def get_total_vectors(self) -> int:
        if not self.files_exist():
            return 0

        state = self.get_faiss_index_state()
        return state.index.ntotal

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> list[DatasetSearchResult]:
        try:
            state = self.get_faiss_index_state()
        except RuntimeError:
            return []

        norm = np.linalg.norm(query_embedding)

        if norm == 0:
            return []

        query_embedding_norm = (query_embedding / norm).astype(np.float32).reshape(1, -1)
        distances, indices = state.index.search(query_embedding_norm, top_k)

        results: list[DatasetSearchResult] = []
        index_metadata = state.metadata

        if indices.size > 0:
            for i, idx_val in enumerate(indices[0]):
                if idx_val != -1 and idx_val < len(index_metadata):
                    similarity_score = 1 - (distances[0][i] ** 2) / 2
                    results.append(
                        DatasetSearchResult(
                            metadata=index_metadata[idx_val],
                            similarity=float(similarity_score),
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
