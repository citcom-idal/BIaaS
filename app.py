import logging
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from biaas.api_query_agent import APIQueryAgent
from biaas.dataset import analyze_dataset, load_dataset_from_bytes
from biaas.dataset.validator import validate_dataset_relevance
from biaas.dataset.visualizer import plot
from biaas.faiss_index import FAISSIndex, build_and_save_index
from biaas.llm.interpreter import generate_insights
from biaas.llm.models import (
    EMBEDDING_MODEL,
)
from biaas.llm.visualizer import (
    gemini_client,
    groq_client,
    plan_visualizations,
)
from biaas.utils import sanitize_filename

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuración inicial ---
load_dotenv()


@st.cache_resource
def get_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource
def get_faiss_index_instance() -> FAISSIndex:
    instance = FAISSIndex()
    # Mover el mensaje de éxito/error a la función main para mejor control del flujo
    return instance


def run_visualization_pipeline(
    user_query: str, df: pd.DataFrame, analysis: dict[str, Any], dataset_title: str
) -> None:
    active_llm_provider = st.session_state.get("current_llm_provider", "gemini")
    st.subheader(f'Analizando consulta (LLM: {active_llm_provider.upper()}): "{user_query}"')
    with st.spinner(f"Generando visualizaciones con {active_llm_provider.upper()}..."):
        df_sample_viz = df.head(20) if len(df) > 20 else df.copy()
        viz_configs_suggested = plan_visualizations(df_sample_viz, user_query, analysis)
        if viz_configs_suggested:
            with st.expander("JSON Sugerencias Visualización", expanded=False):
                st.json(viz_configs_suggested)

    valid_viz_configs_generated = []
    if viz_configs_suggested:
        st.subheader("Visualizaciones sugeridas")
        for idx, config in enumerate(viz_configs_suggested):
            title_viz = config.get("titulo_de_la_visualizacion", f"Visualización {idx+1}")
            st.markdown(f"**{idx+1}. {title_viz}**")
            fig = plot(df, config)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                valid_viz_configs_generated.append(config)
            else:
                st.warning(f"No se pudo generar: {title_viz}")
    else:
        st.info(f"IA ({active_llm_provider.upper()}) no sugirió visualizaciones.")

    with st.spinner(f"Generando insights con {active_llm_provider.upper()}..."):
        st.subheader("💡 Insights del Analista Virtual")
        df_sample_ins = df.head(5) if len(df) > 5 else df.copy()
        insights_text = generate_insights(
            user_query, analysis, valid_viz_configs_generated, df_sample_ins, dataset_title
        )
        st.markdown(insights_text)

    st.success("--- ✅ Análisis completado ---")
    st.balloons()


def main() -> None:
    st.set_page_config(layout="wide", page_title="Analista Datos Valencia")

    if "current_llm_provider" not in st.session_state:
        st.session_state.current_llm_provider = "gemini"
    if "active_df" not in st.session_state:
        st.session_state.active_df = None
    if "active_analysis" not in st.session_state:
        st.session_state.active_analysis = None
    if "active_dataset_title" not in st.session_state:
        st.session_state.active_dataset_title = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "run_initial_analysis" not in st.session_state:
        st.session_state.run_initial_analysis = False

    faiss_index_global = get_faiss_index_instance()
    sentence_model_global = get_sentence_transformer_model(EMBEDDING_MODEL)

    col1, col2 = st.columns([3, 1])
    col1.title("Data València Agent")
    with col2:
        available_llms = [
            llm for llm, client in [("gemini", gemini_client), ("llama3", groq_client)] if client
        ]
        if available_llms:
            st.session_state.current_llm_provider = st.radio(
                "Selecciona LLM:", options=available_llms, horizontal=True, key="llm_selector"
            )
        else:
            st.error("Ningún LLM configurado. Verifica API Keys.")
            st.stop()

    st.sidebar.header("Acciones del Índice")
    if faiss_index_global.is_ready():
        st.sidebar.success(f"Índice FAISS listo ({faiss_index_global.index.ntotal} vectores).")

    if st.sidebar.button("Construir/actualizar Índice FAISS"):
        build_and_save_index()

    if st.session_state.active_df is None:
        display_initial_view(faiss_index_global, sentence_model_global)
    else:
        display_conversation_view()

    st.markdown("---")
    st.caption(
        f"Desarrollado con {EMBEDDING_MODEL} y {st.session_state.get('current_llm_provider','N/A').upper()}."
    )


def display_initial_view(faiss_index: FAISSIndex, sentence_model: SentenceTransformer) -> None:
    st.markdown(
        "Bienvenido al asistente para explorar [Datos Abiertos del Ayuntamiento de Valencia](https://valencia.opendatasoft.com/pages/home/?flg=es-es)."
    )

    if not faiss_index.is_ready():
        st.warning(
            "El índice de búsqueda no está listo. Por favor, constrúyelo desde el menú de la izquierda para poder analizar consultas."
        )
        return

    st.markdown("##### ¿No sabes qué preguntar? Prueba con esto:")
    examples = [
        "Aparcamientos para bicis",
        "Intensidad del tráfico en Valencia",
        "Calidad del aire en la ciudad",
        "Centros educativos",
    ]

    if "user_query_main" not in st.session_state:
        st.session_state.user_query_main = ""

    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        if cols[i].button(example):
            st.session_state.user_query_main = example
            st.rerun()

    user_query_input = st.text_input("¿Qué datos te gustaría analizar?", key="user_query_main")

    if st.button("Analizar Consulta", type="primary"):
        if user_query_input:
            api_agent = APIQueryAgent(faiss_index, sentence_model)
            with st.spinner("Buscando y validando datasets..."):
                search_results = api_agent.search_dataset(user_query_input, top_k=5)
                valid_candidates = []
                if search_results:
                    for result in search_results:
                        if result[
                            "similarity"
                        ] > api_agent.SIMILARITY_THRESHOLD and validate_dataset_relevance(
                            user_query_input,
                            result["metadata"]["title"],
                            result["metadata"]["description"],
                        ):
                            valid_candidates.append(result)

            if not valid_candidates:
                st.error(
                    "No se encontró ningún dataset relevante para tu consulta. Intenta ser más específico o prueba con otra pregunta."
                )
                if search_results:
                    with st.expander(
                        "Resultados de búsqueda con baja relevancia (para depuración):"
                    ):
                        st.json(
                            [
                                {"title": r["metadata"]["title"], "similarity": r["similarity"]}
                                for r in search_results
                            ]
                        )
                return

            selected_dataset_info = sorted(
                valid_candidates, key=lambda x: x["similarity"], reverse=True
            )[0]
            dataset_id = selected_dataset_info["metadata"]["id"]
            dataset_title = selected_dataset_info["metadata"]["title"]

            with st.spinner(f"Descargando y analizando '{dataset_title}'..."):
                dataset_bytes = api_agent.export_dataset(dataset_id)
                if not dataset_bytes:
                    st.error(f"Fallo en la descarga del dataset '{dataset_title}'.")
                    return
                df = load_dataset_from_bytes(dataset_bytes, dataset_title)
                if df is None or df.empty:
                    st.error(f"El dataset '{dataset_title}' está vacío o no se pudo cargar.")
                    return
                analysis = analyze_dataset(df)

            st.session_state.active_df = df
            st.session_state.active_analysis = analysis
            st.session_state.active_dataset_title = dataset_title
            st.session_state.last_query = user_query_input
            st.session_state.run_initial_analysis = True
            st.rerun()
        else:
            st.warning("Por favor, introduce una consulta.")


def display_conversation_view() -> None:
    st.success(f"Dataset activo: **{st.session_state.active_dataset_title}**")
    csv_data = st.session_state.active_df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        label="📥 Descargar Dataset (CSV)",
        data=csv_data,
        file_name=f"{sanitize_filename(st.session_state.active_dataset_title)}.csv",
    )

    if st.session_state.run_initial_analysis:
        run_visualization_pipeline(
            st.session_state.last_query,
            st.session_state.active_df,
            st.session_state.active_analysis,
            st.session_state.active_dataset_title,
        )
        st.session_state.run_initial_analysis = False

    st.markdown("---")
    follow_up_query = st.text_input(
        "Haz una pregunta de seguimiento sobre este dataset:", key="follow_up_query"
    )

    col_run, col_reset = st.columns([3, 1])
    if col_run.button("Analizar Seguimiento", type="primary"):
        if follow_up_query:
            run_visualization_pipeline(
                follow_up_query,
                st.session_state.active_df,
                st.session_state.active_analysis,
                st.session_state.active_dataset_title,
            )
        else:
            st.warning("Introduce una consulta de seguimiento.")

    if col_reset.button("Finalizar y empezar de nuevo"):
        keys_to_delete = [k for k in st.session_state.keys() if k not in ["current_llm_provider"]]
        for key in keys_to_delete:
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
