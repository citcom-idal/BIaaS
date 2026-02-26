import atexit
import logging
from typing import Any

import pandas as pd
import streamlit as st

from app.core.config import settings
from app.core.constants import EMBEDDING_MODEL
from app.core.container import Container
from app.core.exceptions import (
    DatasetNotFoundError,
    ExternalAPIError,
    PlannerError,
    PlannerJSONError,
    PlotGenerationError,
)
from app.llm import LLMModel
from app.schemas.dataset import DatasetSearchResult
from app.services.analysis_service import analyze_dataset
from app.services.dataset_service import DatasetService
from app.services.faiss_index_service import FaissIndexService
from app.services.insights_service import generate_insights
from app.services.plot_service import plot_dataset
from app.services.visual_planner_service import suggest_visualizations
from app.utils import sanitize_filename

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@st.cache_resource
def get_container() -> Container:
    container = Container()

    faiss_index_service = container.faiss_index_service()

    atexit.register(faiss_index_service.shutdown)

    return container


def run_visualization_pipeline(
    llm_model: LLMModel,
    user_query: str,
    df: pd.DataFrame,
    analysis: dict[str, Any],
    dataset_title: str,
) -> None:
    active_llm_provider = settings.LLM_PROVIDER.value
    st.subheader(f'Analizando consulta (LLM: {active_llm_provider.upper()}): "{user_query}"')
    with st.spinner(f"Generando visualizaciones con {active_llm_provider.upper()}..."):
        df_sample_viz = df.head(20) if len(df) > 20 else df.copy()
        viz_configs_suggested: list[dict[str, Any]] = []
        try:
            viz_configs_suggested = suggest_visualizations(
                llm_model=llm_model,
                df_sample=df_sample_viz,
                query=user_query,
                analysis=analysis,
            )
        except PlannerError as e:
            st.error(str(e))
        except PlannerJSONError as e:
            st.error(str(e))
            st.text_area("Respuesta:", e.raw_content, height=150)

        if viz_configs_suggested:
            with st.expander("JSON Sugerencias Visualización", expanded=False):
                st.json(viz_configs_suggested)

    valid_viz_configs_generated = []
    if viz_configs_suggested:
        st.subheader("Visualizaciones sugeridas")
        for idx, config in enumerate(viz_configs_suggested):
            title_viz = config.get("titulo_de_la_visualizacion", f"Visualización {idx + 1}")
            st.markdown(f"**{idx + 1}. {title_viz}**")
            try:
                fig = plot_dataset(df, config)
                st.plotly_chart(fig, use_container_width=True)
                valid_viz_configs_generated.append(config)
            except PlotGenerationError as e:
                if e.level == "warning":
                    st.warning(str(e))
                else:
                    st.error(str(e))

                st.warning(f"No se pudo generar: {title_viz}")
    else:
        st.info(f"IA ({active_llm_provider.upper()}) no sugirió visualizaciones.")

    with st.spinner(f"Generando insights con {active_llm_provider.upper()}..."):
        st.subheader("💡 Insights del Analista Virtual")
        df_sample_ins = df.head(5) if len(df) > 5 else df.copy()
        insights_text = generate_insights(
            llm_model=llm_model,
            query=user_query,
            viz_configs_generated=valid_viz_configs_generated,
            df_sample=df_sample_ins,
            dataset_title=dataset_title,
        )
        st.markdown(insights_text)

    st.success("--- ✅ Análisis completado ---")
    st.balloons()


def main() -> None:
    st.set_page_config(layout="wide", page_title="Analista Datos Valencia")

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

    container = get_container()

    faiss_index_service = container.faiss_index_service()
    dataset_service = container.dataset_service()
    llm_model = container.llm_model_selector()

    faiss_index_service.ensure_index_loaded(timeout=60.0)

    st.title("Data València Agent")

    if st.session_state.active_df is None:
        display_initial_view(faiss_index_service, dataset_service)
    else:
        display_conversation_view(llm_model)

    st.markdown("---")

    st.caption(f"Desarrollado con {EMBEDDING_MODEL} y {settings.LLM_PROVIDER.value.upper()}.")


def display_initial_view(
    faiss_index_service: FaissIndexService, dataset_service: DatasetService
) -> None:
    st.markdown(
        "Bienvenido al asistente para explorar [Datos Abiertos del Ayuntamiento de Valencia](https://valencia.opendatasoft.com/pages/home/?flg=es-es)."
    )

    if not faiss_index_service.files_exist():
        st.warning(
            "El índice de búsqueda no está listo. Por favor, constrúyelo y recarga la página para poder analizar consultas."
        )
        return

    st.success(
        f"Índice FAISS listo con {faiss_index_service.get_total_vectors()} vectores. ¡Puedes empezar a hacer preguntas sobre los datasets de Valencia!"
    )

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
            with st.spinner("Buscando y validando datasets..."):
                try:
                    search_results = dataset_service.search_dataset(user_query_input, top_k=5)
                except DatasetNotFoundError as e:
                    st.error(str(e))
                    search_results = None

                valid_candidates: list[DatasetSearchResult] = []
                if search_results:
                    for result in search_results:
                        if dataset_service.validate_relevance(
                            query=user_query_input, dataset_search_result=result
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
                                {"title": r.metadata.title, "similarity": r.similarity}
                                for r in search_results
                            ]
                        )
                return

            selected_dataset_info = sorted(
                valid_candidates, key=lambda x: x.similarity, reverse=True
            )[0]
            dataset_id = selected_dataset_info.metadata.id
            dataset_title = selected_dataset_info.metadata.title

            with st.spinner(f"Descargando y analizando '{dataset_title}'..."):
                try:
                    df = dataset_service.load_dataset(dataset_id)
                except ExternalAPIError as e:
                    st.error(str(e))
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


def display_conversation_view(llm_model: LLMModel) -> None:
    st.success(f"Dataset activo: **{st.session_state.active_dataset_title}**")
    csv_data = st.session_state.active_df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        label="📥 Descargar Dataset (CSV)",
        data=csv_data,
        file_name=f"{sanitize_filename(st.session_state.active_dataset_title)}.csv",
    )

    if st.session_state.run_initial_analysis:
        run_visualization_pipeline(
            llm_model,
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
                llm_model,
                follow_up_query,
                st.session_state.active_df,
                st.session_state.active_analysis,
                st.session_state.active_dataset_title,
            )
        else:
            st.warning("Introduce una consulta de seguimiento.")

    if col_reset.button("Finalizar y empezar de nuevo"):
        keys_to_delete = list(st.session_state.keys())
        for key in keys_to_delete:
            del st.session_state[key]
        st.rerun()
