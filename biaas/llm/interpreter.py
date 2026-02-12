from typing import Any

import pandas as pd

from biaas.utils import make_llm_call


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

    raw_content = make_llm_call(prompt, is_json_output=False)

    return raw_content if not raw_content.startswith("Error") else "No se generaron insights."
