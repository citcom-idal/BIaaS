from typing import Annotated

from dependency_injector.wiring import Provide

from app.core.container import Container
from app.llm import LLMModel
from app.services.dataset_service import DatasetService
from app.services.faiss_index_service import FaissIndexService

LLMModelDep = Annotated[LLMModel, Provide[Container.llm_model_selector]]
DatasetServiceDep = Annotated[DatasetService, Provide[Container.dataset_service]]
FaissIndexServiceDep = Annotated[FaissIndexService, Provide[Container.faiss_index_service]]
