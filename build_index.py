#!/usr/bin/env python3

from typing import Any

import httpx
import numpy as np
from bs4 import BeautifulSoup
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from sentence_transformers import SentenceTransformer

from app.core.config import (
    CATALOG_LIST_URL,
    EMBEDDING_MODEL,
    INDEX_FILE,
)
from app.core.exceptions import ExternalAPIError
from app.schemas.dataset import DatasetMetadata
from app.services.faiss_index_service import FaissIndexService
from app.utils import fetch


def print_info(message: str) -> None:
    print(message)


def print_success(message: str) -> None:
    print(f"[bold green]{message}[/bold green]")


def print_warning(message: str) -> None:
    print(f"[bold yellow]{message}[/bold yellow]")


def print_error(message: str) -> None:
    print(f"[bold red]{message}[/bold red]")


def generate_dataset_embeddings(
    datasets: Any, sentence_transformer: SentenceTransformer
) -> tuple[list[np.ndarray], list[DatasetMetadata]]:
    texts_for_page: list[str] = []
    embeddings: list[np.ndarray] = []
    embeddings_metadata: list[DatasetMetadata] = []

    for dataset_info in datasets:
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

        dataset_metadata = DatasetMetadata(id=dataset_id, title=title, description=description)

        texts_for_page.append(f"título: {title}; descripción: {description}")
        embeddings_metadata.append(dataset_metadata)

    if texts_for_page:
        page_embeddings = sentence_transformer.encode(
            texts_for_page,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings.extend(page_embeddings)

    return embeddings, embeddings_metadata


def fetch_total_datasets() -> int:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description="Obteniendo catálogo de datasets de OpenData Valencia...", total=None
        )
        params = httpx.QueryParams(limit=1, offset=0)

        try:
            response = fetch(CATALOG_LIST_URL, params=params, timeout=httpx.Timeout(20.0))
        except ExternalAPIError as e:
            progress.stop()
            print_error(f"Error crítico al conectar con la API de OpenData Valencia: {e}")
            return 0

        data = response.json()

        return data.get("total_count", 0)


def fetch_datasets_page(
    total_datasets: int, sentence_transformer: SentenceTransformer
) -> tuple[list[np.ndarray], list[DatasetMetadata]]:
    dataset_embeddings: list[np.ndarray] = []
    dataset_metadatas: list[DatasetMetadata] = []
    start = 0
    limit = 100

    with Progress(transient=True) as progress:
        fetch_datasets_task = progress.add_task(
            description="Procesando datasets y generando embeddings...",
            total=total_datasets,
        )

        while start < total_datasets:
            params = httpx.QueryParams(limit=limit, offset=start)
            try:
                response = fetch(CATALOG_LIST_URL, params=params, timeout=httpx.Timeout(30.0))
            except ExternalAPIError as e:
                progress.stop()
                print_error(f"Error al obtener página de datasets con offset={start}: {e}")
                break

            if response.status_code != 200:
                print_warning(
                    f"Respuesta de la API para offset={start}: Código de estado {response.status_code}. Saltando página."
                )
                start += limit
                continue

            data = response.json()
            datasets_page = data.get("datasets", [])

            if not datasets_page:
                progress.stop()
                print_warning(
                    f"La página con offset={start} no devolvió datasets. Finalizando bucle."
                )
                break

            embeddings, metadata = generate_dataset_embeddings(
                datasets=datasets_page,
                sentence_transformer=sentence_transformer,
            )

            dataset_embeddings.extend(embeddings)
            dataset_metadatas.extend(metadata)

            start += len(datasets_page)
            progress.update(
                fetch_datasets_task,
                advance=len(datasets_page),
                description=f"Procesados {start}/{total_datasets} datasets",
            )

    return dataset_embeddings, dataset_metadatas


def main() -> None:
    sentence_transformer = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    print_info(f"Construyendo Índice FAISS para: {EMBEDDING_MODEL}")

    try:
        total_datasets = fetch_total_datasets()
    except ExternalAPIError as e:
        print_error(f"Error crítico al conectar con la API de OpenData Valencia: {e}")
        return

    if total_datasets == 0:
        print_warning("No se encontraron datasets en el catálogo.")
        return

    print_info(f"Se encontraron {total_datasets} datasets. Procediendo a generar embeddings...")

    try:
        dataset_embeddings, dataset_metadatas = fetch_datasets_page(
            total_datasets=total_datasets,
            sentence_transformer=sentence_transformer,
        )
    except ExternalAPIError as e:
        print_error(f"Error crítico al conectar con la API de OpenData Valencia: {e}")
        return

    if len(dataset_embeddings) == 0:
        print_error(
            "No se pudieron generar embeddings. El proceso ha fallado. Revisa los mensajes de estado de la API."
        )
        return

    faiss_index_service = FaissIndexService()
    print_info(f"Construyendo y guardando el índice FAISS en {INDEX_FILE}...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"Construyendo y guardando el índice FAISS en {INDEX_FILE}...", total=None
        )

        ntotal = faiss_index_service.process_and_save_index(
            dataset_embeddings=dataset_embeddings,
            dataset_metadatas=dataset_metadatas,
        )

        progress.stop()

    print_success(f"¡Índice FAISS con {ntotal} vectores construido y guardado con éxito!")


if __name__ == "__main__":
    main()
