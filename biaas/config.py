from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = str(SCRIPT_DIR / "faiss_opendata_valencia.idx")
METADATA_FILE = str(SCRIPT_DIR / "faiss_metadata.json")

BASE_URL = "https://valencia.opendatasoft.com/api/explore/v2.1/"
CATALOG_LIST_URL = "https://valencia.opendatasoft.com/api/v2/catalog/datasets"
