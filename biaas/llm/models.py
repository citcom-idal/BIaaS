import abc
import os
import re

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)
from groq import Groq

EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
GOOGLE_LLM_MODEL = "gemini-1.5-flash-latest"
LLAMA3_70B_MODEL_NAME_GROQ = "llama3-70b-8192"
GROQ_API_KEY = os.getenv("API_KEY_GROQ", None)
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI", None)


class LLMModel(abc.ABC):
    @abc.abstractmethod
    def get_response(self, prompt: str, json_output: bool = False) -> str:
        pass


class GroqLLMModel(LLMModel):
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def get_response(self, prompt: str, json_output: bool = False) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLAMA3_70B_MODEL_NAME_GROQ,
            temperature=0.1 if json_output else 0.4,
            max_tokens=2048 if json_output else 450,
            response_format={"type": "json_object"} if json_output else None,
        )

        content = chat_completion.choices[0].message.content

        if not content:
            content = ""

        content = content.strip()

        if json_output and content.startswith("```json"):
            match = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
            if match:
                content = match.group(1).strip()

        return content


class GeminiLLMModel(LLMModel):
    def __init__(self):
        self.client = genai.Client(api_key=API_KEY_GEMINI)
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

    def get_response(self, prompt: str, json_output: bool = False) -> str:
        if json_output:
            config = GenerateContentConfig(
                safety_settings=self.safety_settings,
                temperature=0.1,
                max_output_tokens=2048,
                response_mime_type="application/json",
            )
        else:
            config = GenerateContentConfig(
                safety_settings=self.safety_settings,
                temperature=0.4,
                max_output_tokens=450,
            )

        response = self.client.models.generate_content(
            model=GOOGLE_LLM_MODEL,
            contents=prompt,
            config=config,
        )

        if not response.candidates:
            return f"Error Gemini: No candidates. Feedback: {response.prompt_feedback}"

        content = response.candidates[0].content

        if not content or not content.parts:
            return f"Error Gemini: Respuesta sin contenido. Feedback: {response.prompt_feedback}"

        content_parts_text = "".join(p.text for p in content.parts if p.text).strip()

        final_text = (
            response.text if response.text and response.text.strip() else content_parts_text
        )

        if not final_text:
            return "Error Gemini: Respuesta vacía."

        return final_text


def get_llm_provider(llm_provider: str) -> LLMModel:
    if llm_provider == "gemini":
        return GeminiLLMModel()

    if llm_provider == "llama3":
        return GroqLLMModel()

    raise ValueError(f"Error: Proveedor de LLM '{llm_provider}' no reconocido.")
