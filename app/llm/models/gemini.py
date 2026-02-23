from typing import Any

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)
from google.genai.types import GenerateContentConfigOrDict as GeminiConfig

from app.core.config import settings
from app.core.exceptions import LLMModelError
from app.llm.models.base import LLMModel


class GeminiLLMModel(LLMModel):
    def __init__(self) -> None:
        self.safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        self.client = genai.Client(api_key=settings.LLM_PROVIDER_API_KEY)

    def __run_query(self, prompt: str, config: GeminiConfig) -> str:
        try:
            response = self.client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            raise LLMModelError(f"Error Gemini: {e}")

        if not response.candidates:
            raise LLMModelError(
                f"Error Gemini: No candidates. Feedback: {response.prompt_feedback}"
            )

        content = response.candidates[0].content

        if not content or not content.parts:
            raise LLMModelError(
                f"Error Gemini: Respuesta sin contenido. Feedback: {response.prompt_feedback}"
            )

        content_parts_text = "".join(p.text for p in content.parts if p.text).strip()

        final_text = (
            response.text if response.text and response.text.strip() else content_parts_text
        )

        return self._check_content(final_text, "Gemini")

    def get_raw_response(self, prompt: str) -> str:
        return self.__run_query(
            prompt,
            config=GenerateContentConfig(
                safety_settings=self.safety_settings,
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )

    def get_json_response(self, prompt: str) -> tuple[str, Any]:
        content = self.__run_query(
            prompt,
            config=GenerateContentConfig(
                safety_settings=self.safety_settings,
                temperature=0.1,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )

        return content, self._format_json(content)
