from dependency_injector import containers, providers
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.constants import EMBEDDING_MODEL
from app.llm.models import GeminiLLMModel, GroqLLMModel, OllamaLLMModel
from app.services.dataset_service import DatasetService
from app.services.faiss_index_service import FaissIndexService


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["app"])

    llm_model_selector = providers.Selector(
        providers.Object(settings.LLM_PROVIDER),
        gemini=providers.Singleton(GeminiLLMModel),
        groq=providers.Singleton(GroqLLMModel),
        ollama=providers.Singleton(OllamaLLMModel),
    )

    sentence_transformer = providers.Singleton(
        SentenceTransformer,
        model_name_or_path=EMBEDDING_MODEL,
        device="cpu",
    )

    faiss_index_service = providers.Singleton(FaissIndexService)

    dataset_service = providers.Singleton(
        DatasetService,
        sentence_transformer=sentence_transformer,
        faiss_index_service=faiss_index_service,
        llm_model=llm_model_selector,
    )
