from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"

INDEX_FILE = DATA_DIR / "faiss_opendata_valencia.idx"
METADATA_FILE = DATA_DIR / "faiss_metadata.json"

BASE_URL = "https://valencia.opendatasoft.com/api/explore/v2.1/"
CATALOG_LIST_URL = "https://valencia.opendatasoft.com/api/v2/catalog/datasets"

EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"

DATASET_SIMILARITY_THRESHOLD = 0.45
