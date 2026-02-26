import os
import sys

import httpx

BASE_URL = os.environ.get("STREAMLIT_SERVER_BASE_URL_PATH", "")
url = f"http://localhost:8501{BASE_URL}/healthz"

try:
    r = httpx.get(url)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
