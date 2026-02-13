import abc
import json
import re
from typing import Any

from biaas.core.exceptions import LLMModelError


class LLMModel(abc.ABC):
    def _format_json(self, content: str) -> Any:
        match_block = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)

        cleaned_json = match_block.group(1).strip() if match_block else content.strip()

        return json.loads(cleaned_json)

    def _check_content(self, content: str, provider: str) -> str:
        if not content:
            raise LLMModelError(f"Error {provider}: Respuesta vacía.")

        return content.strip()

    @abc.abstractmethod
    def get_raw_response(self, prompt: str) -> str:
        pass

    @abc.abstractmethod
    def get_json_response(self, prompt: str) -> tuple[str, Any]:
        pass
