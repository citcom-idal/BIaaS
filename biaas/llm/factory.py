from biaas.core.config import LLMProvider, settings
from biaas.llm.models.base import LLMModel
from biaas.llm.models.gemini import GeminiLLMModel
from biaas.llm.models.groq import GroqLLMModel
from biaas.llm.models.ollama import OllamaLLMModel

_registry: dict[LLMProvider, type[LLMModel]] = {
    LLMProvider.GEMINI: GeminiLLMModel,
    LLMProvider.GROQ: GroqLLMModel,
    LLMProvider.OLLAMA: OllamaLLMModel,
}


def get_llm_model() -> LLMModel:
    llm_provider = settings.LLM_PROVIDER

    if llm_provider not in _registry:
        raise ValueError(f"Error: Proveedor de LLM '{llm_provider}' no reconocido.")

    return _registry[llm_provider]()
