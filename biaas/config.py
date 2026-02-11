from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


INDEX_FILE = BASE_DIR / "faiss_opendata_valencia.idx"
METADATA_FILE = BASE_DIR / "faiss_metadata.json"

BASE_URL = "https://valencia.opendatasoft.com/api/explore/v2.1/"
CATALOG_LIST_URL = "https://valencia.opendatasoft.com/api/v2/catalog/datasets"

EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    GROQ_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-1.5-flash-latest"

    API_KEY_GEMINI: str | None = None
    GROQ_MODEL: str = "llama3-70b-8192"


settings = Settings()
