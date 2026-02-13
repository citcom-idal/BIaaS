import abc
import json
import re
from typing import Any, Literal

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)
from google.genai.types import GenerateContentConfigOrDict as GeminiConfig
from groq import Groq
from groq.types.chat.completion_create_params import (
    ResponseFormat as GroqResponseFormat,
)
from ollama import Client as OllamaClient

from biaas.config import LLMProvider, settings
from biaas.exceptions import LLMModelError


class LLMModel(abc.ABC):
    def _format_json(self, content: str) -> Any:
        match_block = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)

        cleaned_json = match_block.group(1).strip() if match_block else content.strip()

        return json.loads(cleaned_json)

    @abc.abstractmethod
    def get_raw_response(self, prompt: str) -> str:
        pass

    @abc.abstractmethod
    def get_json_response(self, prompt: str) -> tuple[str, Any]:
        pass


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

        if not content:
            raise LLMModelError("Error Groq: Respuesta vacía.")

        return content.strip()

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


class GeminiLLMModel(LLMModel):
    def __init__(self):
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
                model=settings.resolved_llm_model,
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

        if not final_text:
            raise LLMModelError("Error Gemini: Respuesta vacía.")

        return final_text

    def get_raw_response(self, prompt: str, json_output: bool = False) -> str:
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


class OllamaLLMModel(LLMModel):
    def __init__(self):
        self.client = OllamaClient(host=settings.OLLAMA_HOST.encoded_string())

    def __run_query(self, prompt: str, format: Literal["json"] | None) -> str:
        try:
            response = self.client.chat(
                model=settings.resolved_llm_model,
                messages=[{"role": "user", "content": prompt}],
                format=format,
            )
        except Exception as e:
            raise LLMModelError(f"Error Ollama: {e}")

        content = response.message.content

        if not content:
            raise LLMModelError("Error Ollama: Respuesta vacía.")

        return content.strip()

    def get_raw_response(self, prompt: str) -> str:
        return self.__run_query(prompt, format=None)

    def get_json_response(self, prompt) -> tuple[str, Any]:
        content = self.__run_query(prompt, format="json")

        return content, self._format_json(content)


def get_llm_model() -> LLMModel:
    llm_provider = settings.LLM_PROVIDER

    if llm_provider == LLMProvider.GEMINI:
        return GeminiLLMModel()

    if llm_provider == LLMProvider.GROQ:
        return GroqLLMModel()

    if llm_provider == LLMProvider.OLLAMA:
        return OllamaLLMModel()

    raise ValueError(f"Error: Proveedor de LLM '{llm_provider}' no reconocido.")
