from typing import Any

import pandas as pd

from app.core.exceptions import (
    LLMModelError,
    PlannerError,
    PlannerJSONError,
)
from app.llm import LLMModel


def suggest_visualizations(
    llm_model: LLMModel, df_sample: pd.DataFrame, query: str, analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    try:
        df_head_str = df_sample.head(3).to_markdown(index=False)
    except Exception:
        df_head_str = df_sample.head(3).to_string()

    value_counts_summary = "\nValores Comunes en Categóricas (Top 3):\n"

    for col, counts in list(analysis.get("value_counts", {}).items())[:5]:
        top_items = list(counts.items())[:3]
        value_counts_summary += f"- {col}: {', '.join([f'{k} ({v})' for k, v in top_items])}\n"

    prompt = f"""Actúa como un analista de BI experto. Tu objetivo es proponer las mejores visualizaciones para responder a la consulta de un usuario, usando un dataset.
Consulta del Usuario: "{query}"
Dataset Resumido (primeras filas de {df_sample.shape[0]}):
{df_head_str}
Análisis de Columnas (USA ESTOS NOMBRES EXACTOS):
- Numéricas: {analysis["numeric"]}
- Categóricas: {analysis["categorical"]}
- Temporales: {analysis["temporal"]}
- Geoespaciales: {analysis["geospatial"]}
{value_counts_summary if len(analysis.get("value_counts", {})) > 0 else ""}
Instrucciones:
1.  Prioriza la consulta del usuario.
2.  Usa **SOLAMENTE las columnas listadas**. Nombres EXACTOS.
3.  Sugiere entre 2 y 4 visualizaciones variadas.
4.  Formato de Salida: **SOLAMENTE la lista JSON válida** ([{{...}}, {{...}}]).
5.  Para cada visualización, proporciona:
    - "tipo_de_visualizacion": (String) Elige ESTRICTAMENTE de esta lista: ["histograma", "grafico de barras", "mapa de puntos", "grafico de lineas", "grafico circular", "diagrama de caja", "treemap", "mapa de calor"].
    - "campos_involucrados": (Lista de strings) Nombres EXACTOS de columnas.
    - "titulo_de_la_visualizacion": (String) Título descriptivo.
    - "descripcion_utilidad": (String) Qué muestra y cómo ayuda.
    - "plotly_params": (Opcional, Dict) Parámetros para Plotly Express.
Genera el JSON:"""

    try:
        raw_content, visualizations = llm_model.get_json_response(prompt)

        if isinstance(visualizations, dict):
            keys = list(visualizations.keys())

            if len(keys) == 1 and isinstance(visualizations[keys[0]], list):
                visualizations = visualizations[keys[0]]
            else:
                visualizations = [visualizations]

        return visualizations if isinstance(visualizations, list) else []
    except LLMModelError as e:
        raise PlannerError(str(e))
    except Exception:
        raise PlannerJSONError(raw_content)
