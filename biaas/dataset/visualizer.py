import unicodedata
from collections.abc import Callable
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PLOT_FUNCTIONS: dict[str, Callable[..., go.Figure]] = {
    "histograma": px.histogram,
    "barras": px.bar,
    "lineas": px.line,
    "dispersion": px.scatter,
    "caja": px.box,
    "puntos": px.scatter_map,
    "circular": px.pie,
    "treemap": px.treemap,
    "calor": px.density_mapbox,
}


CANONICAL_CHART_TYPES = {
    "barras": ["barras", "bar"],
    "lineas": ["lineas", "line"],
    "puntos": ["mapa de puntos", "scatter map"],
    "calor": ["mapa de calor", "density map"],
    "dispersion": ["dispersion", "scatter"],
    "caja": ["diagrama de caja", "box"],
    "circular": ["circular", "tarta"],
}


def _normalize_chart_type(chart_type: str) -> str:
    if not isinstance(chart_type, str):
        return ""

    nfkd_form = unicodedata.normalize("NFKD", chart_type.lower().strip())
    normalized = "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    for canonical, aliases in CANONICAL_CHART_TYPES.items():
        if any(alias in normalized for alias in aliases):
            return canonical

    return normalized


def plot_dataset(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure | None:
    chart_type = _normalize_chart_type(config.get("tipo_de_visualizacion", ""))
    plot_func = PLOT_FUNCTIONS.get(chart_type)

    if not plot_func:
        st.warning(f"Tipo visualización no soportado: '{config.get('tipo_de_visualizacion', '')}'")
        return None

    campos_orig = config.get("campos_involucrados", [])
    campos = [c for c in campos_orig if c in df.columns]
    if not campos and campos_orig:
        st.warning(f"Campos no existen ({campos_orig}).")
        return None
    if not campos:
        st.warning(f"No hay campos para '{chart_type}'.")
        return None

    title = config.get("titulo_de_la_visualizacion", chart_type)

    df_plot = df.copy()

    plot_args = {"data_frame": df_plot, "title": title, **(config.get("plotly_params", {}) or {})}
    plot_args.pop("type", None)

    try:
        if chart_type == "histograma":
            plot_args["x"] = campos[0]
        elif chart_type == "circular":
            plot_args.update({"names": campos[0], "values": campos[1] if len(campos) > 1 else None})
        elif chart_type in ["barras", "lineas", "dispersion", "caja"]:
            plot_args.update({"x": campos[0], "y": campos[1] if len(campos) > 1 else None})
        elif chart_type == "treemap":
            plot_args["path"] = [c for c in campos if c in df.columns]
        elif chart_type in ["puntos", "calor"]:
            lat_col, lon_col = None, None
            p_lat, p_lon = plot_args.get("lat"), plot_args.get("lon")
            if p_lat in df.columns and p_lon in df.columns:
                lat_col, lon_col = p_lat, p_lon
            elif "latitude" in df.columns and "longitude" in df.columns:
                lat_col, lon_col = "latitude", "longitude"
            else:
                for c in campos:
                    cl = c.lower()
                    if "latit" in cl or cl == "y":
                        lat_col = c
                    if "longit" in cl or cl == "x":
                        lon_col = c
                    if lat_col and lon_col:
                        break
            if not (lat_col and lon_col):
                raise ValueError("No se pudieron encontrar columnas de latitud/longitud.")

            df_plot[lat_col] = pd.to_numeric(df_plot[lat_col], errors="coerce")
            df_plot[lon_col] = pd.to_numeric(df_plot[lon_col], errors="coerce")
            df_plot.dropna(subset=[lat_col, lon_col], inplace=True)
            if df_plot.empty:
                st.warning(
                    "No hay datos geoespaciales válidos para mostrar después de la limpieza."
                )
                return None

            plot_args.update({"lat": lat_col, "lon": lon_col, "zoom": plot_args.get("zoom", 10)})

            if chart_type == "calor":
                plot_args["mapbox_style"] = "open-street-map"
                z_cands = [
                    f
                    for f in campos
                    if f not in [lat_col, lon_col] and pd.api.types.is_numeric_dtype(df[f])
                ]
                if z_cands:
                    plot_args["z"] = z_cands[0]
                    df_plot[z_cands[0]] = pd.to_numeric(df_plot[z_cands[0]], errors="coerce")
                    df_plot.dropna(subset=[z_cands[0]], inplace=True)

        return plot_func(**plot_args)
    except Exception as e:
        st.error(f"Error generando el gráfico '{title}': {e}")
        return None
