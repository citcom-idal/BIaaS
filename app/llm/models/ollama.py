from typing import Any, Literal

from ollama import Client as OllamaClient

from app.core.config import get_settings
from app.core.exceptions import LLMModelError
from app.llm.models.base import LLMModel

settings = get_settings()


class OllamaLLMModel(LLMModel):
    def __init__(self) -> None:
        self.client = OllamaClient(host=settings.OLLAMA_HOST.encoded_string())

    def __run_query(self, prompt: str, format: Literal["json"] | None) -> str:
        try:
            response = self.client.chat(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format=format,
            )
        except Exception as e:
            raise LLMModelError(f"Error Ollama: {e}")

        content = response.message.content

        return self._check_content(content, "Ollama")

    def get_raw_response(self, prompt: str) -> str:
        return self.__run_query(prompt, format=None)

    def get_json_response(self, prompt: str) -> tuple[str, Any]:
        content = self.__run_query(prompt, format="json")

        return content, self._format_json(content)
