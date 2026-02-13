import abc
import re

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)
from groq import Groq
from ollama import Client as OllamaClient

from biaas.config import LLMProvider, settings
from biaas.exceptions import LLMModelError


class LLMModel(abc.ABC):
    @abc.abstractmethod
    def get_response(self, prompt: str, json_output: bool = False) -> str:
        pass


class GroqLLMModel(LLMModel):
    def __init__(self):
        self.client = Groq(api_key=settings.LLM_PROVIDER_API_KEY)

    def get_response(self, prompt: str, json_output: bool = False) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.resolved_llm_model,
                temperature=0.1 if json_output else 0.4,
                max_tokens=2048 if json_output else 450,
                response_format={"type": "json_object"} if json_output else None,
            )
        except Exception as e:
            raise LLMModelError(f"Error Groq: {e}")

        content = chat_completion.choices[0].message.content

        if not content:
            raise LLMModelError("Error Groq: Respuesta vacía.")

        content = content.strip()

        if json_output and content.startswith("```json"):
            match = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
            if match:
                content = match.group(1).strip()

        return content


class GeminiLLMModel(LLMModel):
    def __init__(self):
        safety_settings = [
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
        self.raw_config = GenerateContentConfig(
            safety_settings=safety_settings, temperature=0.1, max_output_tokens=2048
        )
        self.json_config = GenerateContentConfig(
            safety_settings=safety_settings,
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )

    def get_response(self, prompt: str, json_output: bool = False) -> str:
        try:
            response = self.client.models.generate_content(
                model=settings.resolved_llm_model,
                contents=prompt,
                config=self.json_config if json_output else self.raw_config,
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


class OllamaLLMModel(LLMModel):
    def __init__(self):
        self.client = OllamaClient(host=settings.OLLAMA_HOST.encoded_string())

    def get_response(self, prompt: str, json_output: bool = False) -> str:
        try:
            response = self.client.chat(
                model=settings.resolved_llm_model,
                messages=[{"role": "user", "content": prompt}],
                format="json" if json_output else "",
            )
        except Exception as e:
            raise LLMModelError(f"Error Ollama: {e}")

        content = response.message.content

        if not content:
            raise LLMModelError("Error Ollama: Respuesta vacía.")

        return content.strip()


def get_llm_model() -> LLMModel:
    llm_provider = settings.LLM_PROVIDER

    if llm_provider == LLMProvider.GEMINI:
        return GeminiLLMModel()

    if llm_provider == LLMProvider.GROQ:
        return GroqLLMModel()

    if llm_provider == LLMProvider.OLLAMA:
        return OllamaLLMModel()

    raise ValueError(f"Error: Proveedor de LLM '{llm_provider}' no reconocido.")
