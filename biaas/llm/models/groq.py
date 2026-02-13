import re
from typing import Any

from groq import Groq
from groq.types.chat.completion_create_params import (
    ResponseFormat as GroqResponseFormat,
)

from biaas.config import settings
from biaas.exceptions import LLMModelError
from biaas.llm.models.base import LLMModel


class GroqLLMModel(LLMModel):
    def __init__(self):
        self.client = Groq(api_key=settings.LLM_PROVIDER_API_KEY)

    def __run_query(
        self, prompt: str, temperature: float, max_tokens: int, response_format: GroqResponseFormat
    ) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.resolved_llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as e:
            raise LLMModelError(f"Error Groq: {e}")

        content = chat_completion.choices[0].message.content

        return self._check_content(content, "Groq")

    def get_raw_response(self, prompt: str) -> str:
        return self.__run_query(prompt, temperature=0.4, max_tokens=450, response_format=None)

    def get_json_response(self, prompt: str) -> tuple[str, Any]:
        content = self.__run_query(
            prompt, temperature=0.1, max_tokens=2048, response_format={"type": "json_object"}
        )

        if content.startswith("```json"):
            match = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
            if match:
                content = match.group(1).strip()

        return content, self._format_json(content)
