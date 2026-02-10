from biaas.llm.visualizer import make_llm_call


def validate_dataset_relevance(
    llm_provider: str, query: str, dataset_title: str, dataset_description: str
) -> bool:
    prompt = f"""Evalúa la relevancia. Consulta: "{query}". Dataset: Título="{dataset_title}", Desc="{dataset_description[:300]}". ¿Es este dataset ALTAMENTE relevante para la consulta? Responde solo con 'Sí' o 'No'."""
    raw_response = make_llm_call(llm_provider, prompt, is_json_output=False)
    return "sí" in raw_response.lower() or "si" in raw_response.lower()
