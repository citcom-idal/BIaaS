import logging
import os

from google import genai
from google.genai.types import (
    ContentListUnionDict,
    GenerateContentConfig,
    GenerateContentResponse,
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


def get_groq_client() -> Groq:
    if GROQ_API_KEY is None:
        message = "GROQ_API_KEY no está configurada."
        logging.error(message)
        raise ValueError(message)

    return Groq(api_key=GROQ_API_KEY)


def get_groq_response(client: Groq, prompt_text: str, json_output: bool = False) -> str:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_text}],
        model=LLAMA3_70B_MODEL_NAME_GROQ,
        temperature=0.1 if json_output else 0.4,
        max_tokens=2048 if json_output else 450,
        response_format={"type": "json_object"} if json_output else None,
    )

    content = chat_completion.choices[0].message.content

    if not content:
        return ""

    return content.strip()


def get_gemini_model() -> genai.Client:
    if API_KEY_GEMINI is None:
        message = "API_KEY_GEMINI no está configurada."
        logging.error(message)
        raise ValueError(message)

    return genai.Client(api_key=API_KEY_GEMINI)


def get_gemini_response(
    client: genai.Client, contents: ContentListUnionDict, json_output: bool = False
) -> GenerateContentResponse:
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

    if json_output:
        config = GenerateContentConfig(
            safety_settings=safety_settings,
            temperature=0.1,
            max_output_tokens=2048,
            response_mime_type="application/json",
        )
    else:
        config = GenerateContentConfig(
            safety_settings=safety_settings,
            temperature=0.4,
            max_output_tokens=450,
        )

    return client.models.generate_content(
        model=GOOGLE_LLM_MODEL,
        contents=contents,
        config=config,
    )
