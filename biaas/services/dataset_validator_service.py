from biaas.exceptions import LLMModelError
from biaas.llm import get_llm_model


def validate_dataset_relevance(query: str, dataset_title: str, dataset_description: str) -> bool:
    prompt = f"""Evalúa la relevancia. Consulta: "{query}". Dataset: Título="{dataset_title}", Desc="{dataset_description[:300]}". ¿Es este dataset ALTAMENTE relevante para la consulta? Responde solo con 'Sí' o 'No'."""

    try:
        llm_model = get_llm_model()
        raw_response = llm_model.get_raw_response(prompt)
    except LLMModelError:
        return False

    return "sí" in raw_response.lower() or "si" in raw_response.lower()
