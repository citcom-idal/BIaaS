import enum
from functools import lru_cache
from typing import Self

from pydantic import Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import BASE_DIR


class LLMProvider(enum.StrEnum):
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

    ENVIRONMENT: str = "development"

    LLM_PROVIDER: LLMProvider = Field(default=...)
    LLM_MODEL: str = Field(default=...)

    LLM_PROVIDER_API_KEY: str | None = None
    OLLAMA_HOST: HttpUrl = HttpUrl("http://localhost:11434")

    @model_validator(mode="after")
    def _validate_provider(self) -> Self:
        if self.LLM_PROVIDER == LLMProvider.GROQ and not self.LLM_PROVIDER_API_KEY:
            raise ValueError("GROQ_API_KEY must be set when LLM_PROVIDER is 'groq'")

        if self.LLM_PROVIDER == LLMProvider.GEMINI and not self.LLM_PROVIDER_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set when LLM_PROVIDER is 'gemini'")

        return self

    USE_EMBEDDING_MODEL_CACHE: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
