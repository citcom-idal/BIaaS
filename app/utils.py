import re

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.exceptions import ExternalAPIError


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception_type(httpx.RequestError),
    reraise=True,
)
def fetch_with_retry(
    url: str, params: httpx.QueryParams | None = None, timeout: httpx.Timeout = httpx.Timeout(5.0)
) -> httpx.Response:
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, params=params)
        response.raise_for_status()

        return response


def fetch(
    url: str,
    params: httpx.QueryParams | None = None,
    timeout: httpx.Timeout = httpx.Timeout(5.0),
) -> httpx.Response:
    with httpx.Client(timeout=timeout) as client:
        try:
            response = client.get(url, params=params)
            response.raise_for_status()

            return response
        except httpx.TimeoutException as e:
            raise ExternalAPIError(f"Timeout al acceder a {url}: {e}") from e
        except httpx.RequestError as e:
            raise ExternalAPIError(f"Error de conexión al acceder a {url}: {e}") from e
        except httpx.HTTPStatusError as e:
            raise ExternalAPIError(
                f"Error HTTP {e.response.status_code} al acceder a {url}: {e}"
            ) from e
        except Exception as e:
            raise ExternalAPIError(f"Error inesperado al acceder a {url}: {e}") from e
