from typing import Any

import pandas as pd

from app.core.exceptions import LLMModelError
from app.llm import get_llm_model


def generate_insights(
    query: str,
    viz_configs_generated: list[dict[str, Any]],
    df_sample: pd.DataFrame,
    dataset_title: str,
) -> str:
    if not viz_configs_generated:
        return "No se generaron visualizaciones válidas."

    viz_summary = "\n".join(
        [
            f"- **{c.get('titulo_de_la_visualizacion', 'N/A')}** ({c.get('tipo_de_visualizacion', 'N/A')}): {c.get('descripcion_utilidad', 'N/A')}"
            for c in viz_configs_generated
        ]
    )
    prompt = f"""Actúa como analista de datos conciso.
Contexto:
- Consulta: "{query}"
- Dataset: '{dataset_title}' ({df_sample.shape[0]} filas en muestra).
- Visualizaciones Generadas: {viz_summary}
Tarea: Redacta un resumen breve (1-2 párrafos, máx 120 palabras) con los insights más relevantes. Céntrate en responder la consulta. No inventes información.
Genera el resumen:"""

    try:
        llm_model = get_llm_model()
        return llm_model.get_raw_response(prompt)
    except LLMModelError:
        return "No se generaron insights"
