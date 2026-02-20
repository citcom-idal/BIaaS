import re

import httpx

from app.core.exceptions import ExternalAPIError


def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def fetch_url(
    url: str,
    params: httpx.QueryParams | None = None,
    timeout: httpx.Timeout = httpx.Timeout(5.0),
) -> httpx.Response:
    with httpx.Client(timeout=timeout) as client:
        try:
            response = client.get(url, params=params)
            response.raise_for_status()

            return response
        except httpx.RequestError as e:
            raise ExternalAPIError(f"Error de conexión al acceder a {url}: {e}")
        except httpx.HTTPStatusError as e:
            raise ExternalAPIError(f"Error HTTP {e.response.status_code} al acceder a {url}: {e}")
        except httpx.TimeoutException as e:
            raise ExternalAPIError(f"Timeout al acceder a {url}: {e}")
        except Exception as e:
            raise ExternalAPIError(f"Error inesperado al acceder a {url}: {e}")
