import logging
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from app.core.config import (
    EMBEDDING_MODEL,
    INDEX_FILE,
    settings,
)
from app.core.exceptions import (
    DatasetNotFoundError,
    ExternalAPIError,
    PlannerError,
    PlannerJSONError,
    PlotGenerationError,
)
from app.schemas.dataset import DatasetMetadata, DatasetSearchResult
from app.services.analysis_service import analyze_dataset
from app.services.api_query_agent_service import APIQueryAgent
from app.services.dataset_service import DatasetService
from app.services.faiss_service import FaissService
from app.services.insights_service import generate_insights
from app.services.plot_service import plot_dataset
from app.services.visual_planner_service import suggest_visualizations
from app.utils import sanitize_filename

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@st.cache_resource
def get_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource
def get_faiss_index_instance() -> FaissService:
    instance = FaissService()
    # Mover el mensaje de éxito/error a la función main para mejor control del flujo
    return instance


def build_and_save_index() -> None:
    faiss_service = get_faiss_index_instance()
    sentence_model_instance = get_sentence_transformer_model(EMBEDDING_MODEL)
    api_query_agent = APIQueryAgent()
    dataset_service = DatasetService(sentence_model_instance, faiss_service)

    st.header(f"Construyendo Índice FAISS para: {EMBEDDING_MODEL}")

    dataset_embeddings: list[np.ndarray] = []
    dataset_metadatas: list[DatasetMetadata] = []

    limit = 100
    start = 0
    total_datasets = None

    with st.spinner("Obteniendo catálogo de datasets de OpenData Valencia..."):
        try:
            total_datasets = api_query_agent.fetch_datasets_count()

            if total_datasets == 0:
                st.warning("No se encontraron datasets en el catálogo.")
                return

            st.info(
                f"Se encontraron {total_datasets} datasets. Procediendo a generar embeddings..."
            )
        except ExternalAPIError as e:
            st.error(f"Error crítico al conectar con la API de OpenData Valencia: {e}")
            return

    progress_bar = st.progress(0.0, "Iniciando proceso...")
    status_area = st.empty()

    while start < total_datasets:
        try:
            status_area.write(f"Obteniendo página de datasets... offset={start}, limit={limit}")
            response = api_query_agent.fetch_datasets_page(limit=limit, offset=start)

            if response.status_code != 200:
                status_area.warning(
                    f"Respuesta de la API para offset={start}: Código de estado {response.status_code}. Saltando página."
                )
                start += limit
                continue

            data = response.json()
            datasets_page = data.get("datasets", [])

            if not datasets_page:
                status_area.warning(
                    f"La página con offset={start} no devolvió datasets. Finalizando bucle."
                )
                break

            embeddings, metadata = dataset_service.generate_dataset_embeddings(datasets_page)

            dataset_embeddings.extend(embeddings)
            dataset_metadatas.extend(metadata)

            start += len(datasets_page)
            progress_bar.progress(
                min(start / total_datasets, 1.0),
                text=f"Procesados {start}/{total_datasets} datasets",
            )

        except ExternalAPIError as e:
            st.error(str(e))
            break

    if not dataset_embeddings:
        st.error(
            "No se pudieron generar embeddings. El proceso ha fallado. Revisa los mensajes de estado de la API."
        )
        return

    progress_bar.empty()
    status_area.empty()

    st.info(f"Construyendo y guardando el índice FAISS en {INDEX_FILE}...")

    with st.spinner(f"Construyendo y guardando el índice FAISS en {INDEX_FILE}..."):
        saved_vector_count = faiss_service.process_and_save_index(
            dataset_embeddings, dataset_metadatas
        )

    st.success(f"¡Índice FAISS con {saved_vector_count} vectores construido y guardado con éxito!")
    st.balloons()
    st.info("La página se recargará para usar el nuevo índice.")
    st.rerun()


def run_visualization_pipeline(
    user_query: str, df: pd.DataFrame, analysis: dict[str, Any], dataset_title: str
) -> None:
    active_llm_provider = settings.LLM_PROVIDER.value
    st.subheader(f'Analizando consulta (LLM: {active_llm_provider.upper()}): "{user_query}"')
    with st.spinner(f"Generando visualizaciones con {active_llm_provider.upper()}..."):
        df_sample_viz = df.head(20) if len(df) > 20 else df.copy()
        viz_configs_suggested: list[dict[str, Any]] = []
        try:
            viz_configs_suggested = suggest_visualizations(df_sample_viz, user_query, analysis)
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
            title_viz = config.get("titulo_de_la_visualizacion", f"Visualización {idx+1}")
            st.markdown(f"**{idx+1}. {title_viz}**")
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
            user_query,
            valid_viz_configs_generated,
            df_sample_ins,
            dataset_title,
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

    faiss_index_global = get_faiss_index_instance()
    sentence_model_global = get_sentence_transformer_model(EMBEDDING_MODEL)
    dataset_service = DatasetService(sentence_model_global, faiss_index_global)

    faiss_index_global.load_index()

    st.title("Data València Agent")

    st.sidebar.header("Acciones del Índice")
    if faiss_index_global.is_ready():
        st.sidebar.success(f"Índice FAISS listo ({faiss_index_global.index.ntotal} vectores).")

    if st.sidebar.button("Construir/actualizar Índice FAISS"):
        build_and_save_index()

    if st.session_state.active_df is None:
        display_initial_view(faiss_index_global, dataset_service)
    else:
        display_conversation_view()

    st.markdown("---")

    st.caption(f"Desarrollado con {EMBEDDING_MODEL} y {settings.LLM_PROVIDER.value.upper()}.")


def display_initial_view(
    faiss_service: FaissService,
    dataset_service: DatasetService,
) -> None:
    st.markdown(
        "Bienvenido al asistente para explorar [Datos Abiertos del Ayuntamiento de Valencia](https://valencia.opendatasoft.com/pages/home/?flg=es-es)."
    )

    if not faiss_service.is_ready():
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
            api_agent = APIQueryAgent()
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
                    df = api_agent.load_dataset(dataset_id)
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
        keys_to_delete = list(st.session_state.keys())
        for key in keys_to_delete:
            del st.session_state[key]
        st.rerun()
