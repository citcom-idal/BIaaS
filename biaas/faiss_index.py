import json
import logging
import os
from typing import Any

import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup

from app import get_sentence_transformer_model
from biaas.config import CATALOG_LIST_URL, EMBEDDING_MODEL, INDEX_FILE, METADATA_FILE


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


def build_and_save_index(
    target_model_name: str = EMBEDDING_MODEL,
    index_path_to_save: str = INDEX_FILE,
    metadata_path_to_save: str = METADATA_FILE,
) -> None:
    st.header(f"Construyendo Índice FAISS para: {target_model_name}")

    try:
        sentence_model_instance = get_sentence_transformer_model(target_model_name)
    except Exception as e:
        st.error(f"Error al cargar el modelo de embeddings: {e}")
        return

    all_metadata: list[Any] = []
    all_embeddings_list: list[Any] = []

    limit = 100
    start = 0
    total_datasets = None

    with st.spinner("Obteniendo catálogo de datasets de OpenData Valencia..."):
        try:
            initial_response = requests.get(
                CATALOG_LIST_URL, params={"limit": 1, "offset": 0}, timeout=20
            )
            initial_response.raise_for_status()
            total_datasets = initial_response.json().get("total_count", 0)
            if total_datasets == 0:
                st.warning("No se encontraron datasets en el catálogo.")
                return
            st.info(
                f"Se encontraron {total_datasets} datasets. Procediendo a generar embeddings..."
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Error crítico al conectar con la API de OpenData Valencia: {e}")
            return

    progress_bar = st.progress(0.0, "Iniciando proceso...")
    status_area = st.empty()

    while start < total_datasets:
        try:
            status_area.write(f"Obteniendo página de datasets... offset={start}, limit={limit}")
            params = {"limit": limit, "offset": start}
            response = requests.get(CATALOG_LIST_URL, params=params, timeout=30)

            if response.status_code != 200:
                status_area.warning(
                    f"Respuesta de la API para offset={start}: Código de estado {response.status_code}. Saltando página."
                )
                start += limit
                continue

            data = response.json()
            datasets_page = data.get("datasets", [])

            if not datasets_page:
                status_area.warning(
                    f"La página con offset={start} no devolvió datasets. Finalizando bucle."
                )
                break

            texts_for_page = []
            metadata_for_page = []

            for dataset_info in datasets_page:
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

                texts_for_page.append(f"título: {title}; descripción: {description}")
                metadata_for_page.append(
                    {"id": dataset_id, "title": title, "description": description}
                )

            if texts_for_page:
                page_embeddings = sentence_model_instance.encode(
                    texts_for_page, normalize_embeddings=True, show_progress_bar=False
                )
                all_embeddings_list.extend(page_embeddings)
                all_metadata.extend(metadata_for_page)

            start += len(datasets_page)
            progress_bar.progress(
                min(start / total_datasets, 1.0),
                text=f"Procesados {start}/{total_datasets} datasets",
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Error de red en offset {start}. Deteniendo... Error: {e}")
            break
        except Exception as e:
            st.error(f"Ocurrió un error inesperado en offset {start}: {e}")
            break

    if not all_embeddings_list:
        st.error(
            "No se pudieron generar embeddings. El proceso ha fallado. Revisa los mensajes de estado de la API."
        )
        return

    progress_bar.empty()
    status_area.empty()

    with st.spinner(f"Construyendo y guardando el índice FAISS en {index_path_to_save}..."):
        embeddings_np = np.array(all_embeddings_list).astype("float32")
        d = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_np)

        faiss.write_index(index, index_path_to_save)

        with open(metadata_path_to_save, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    st.success(f"¡Índice FAISS con {index.ntotal} vectores construido y guardado con éxito!")
    st.balloons()
    st.info("La página se recargará para usar el nuevo índice.")
    st.rerun()
