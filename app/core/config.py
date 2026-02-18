import enum
from pathlib import Path
from typing import Self

from pydantic import HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"

INDEX_FILE = DATA_DIR / "faiss_opendata_valencia.idx"
METADATA_FILE = DATA_DIR / "faiss_metadata.json"

BASE_URL = "https://valencia.opendatasoft.com/api/explore/v2.1/"
CATALOG_LIST_URL = "https://valencia.opendatasoft.com/api/v2/catalog/datasets"

EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"

DATASET_SIMILARITY_THRESHOLD = 0.45


class LLMProvider(enum.Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    LLM_PROVIDER: LLMProvider
    LLM_MODEL: str

    LLM_PROVIDER_API_KEY: str | None = None
    OLLAMA_HOST: HttpUrl = HttpUrl("http://localhost:11434")

    @model_validator(mode="after")
    def _validate_provider(self) -> Self:
        if self.LLM_PROVIDER == LLMProvider.GROQ and not self.LLM_PROVIDER_API_KEY:
            raise ValueError("GROQ_API_KEY must be set when LLM_PROVIDER is 'groq'")

        if self.LLM_PROVIDER == LLMProvider.GEMINI and not self.LLM_PROVIDER_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set when LLM_PROVIDER is 'gemini'")

        return self


settings = Settings()
