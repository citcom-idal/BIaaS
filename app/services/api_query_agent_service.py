import io

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import CATALOG_LIST_URL
from app.core.exceptions import ExternalAPIError


class APIQueryAgent:
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
