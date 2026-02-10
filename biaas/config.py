from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


INDEX_FILE: str = BASE_DIR / "faiss_opendata_valencia.idx"
METADATA_FILE: str = BASE_DIR / "faiss_metadata.json"

BASE_URL: str = "https://valencia.opendatasoft.com/api/explore/v2.1/"
CATALOG_LIST_URL: str = "https://valencia.opendatasoft.com/api/v2/catalog/datasets"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    EMBEDDING_MODEL: str = "paraphrase-MiniLM-L6-v2"
    GOOGLE_LLM_MODEL: str = "gemini-1.5-flash-latest"
    LLAMA3_70B_MODEL_NAME_GROQ: str = "llama3-70b-8192"
    GROQ_API_KEY: str | None = None
    API_KEY_GEMINI: str | None = None


settings = Settings()
