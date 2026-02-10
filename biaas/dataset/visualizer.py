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
    "barras": ["barras", "bar", "bar chart", "histograma"],
    "lineas": ["lineas", "line", "line chart"],
    "dispersion": ["dispersion", "scatter", "scatter plot"],
    "puntos": ["mapa de puntos", "scatter map", "point map"],
    "calor": ["mapa de calor", "heatmap", "density map"],
    "caja": ["diagrama de caja", "box", "boxplot"],
    "circular": ["circular", "tarta", "pie", "pie chart"],
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


def _args_histograma(plot_args: dict[str, Any], fields: list[Any]):
    plot_args["x"] = fields[0]


def _args_xy(plot_args: dict[str, Any], fields: list[Any]):
    plot_args["x"] = fields[0]
    if len(fields) > 1:
        plot_args["y"] = fields[1]


def _args_circular(plot_args: dict[str, Any], fields: list[Any]):
    plot_args["names"] = fields[0]
    if len(fields) > 1:
        plot_args["values"] = fields[1]


def _args_treemap(plot_args: dict[str, Any], fields: list[Any]):
    plot_args["path"] = fields


def _attach_z_if_possible(
    plot_args: dict[str, Any], df: pd.DataFrame, fields: list[Any], lat_col: str, lon_col: str
):
    for c in fields:
        if c not in (lat_col, lon_col) and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=[c], inplace=True)
            plot_args["z"] = c
            break


def _find_lat_lon(df: pd.DataFrame, fields: list[Any], plot_args: dict[str, Any]):
    lat_col, lon_col = plot_args.get("lat"), plot_args.get("lon")
    if lat_col in df.columns and lon_col in df.columns:
        return lat_col, lon_col

    if "latitude" in df.columns and "longitude" in df.columns:
        return "latitude", "longitude"

    for c in fields:
        cl = c.lower()
        if "latit" in cl or cl == "y":
            lat_col = c

        if "longit" in cl or cl == "x":
            lon_col = c

        if lat_col and lon_col:
            return lat_col, lon_col

    raise ValueError("No se pudieron encontrar columnas de latitud/longitud.")


def _args_geo(plot_args: dict[str, Any], df: pd.DataFrame, fields: list[Any], chart_type: str):
    lat_col, lon_col = _find_lat_lon(df, fields, plot_args)

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df.dropna(subset=[lat_col, lon_col], inplace=True)

    if df.empty:
        raise ValueError("No hay datos geoespaciales válidos.")

    plot_args.update(
        {
            "lat": lat_col,
            "lon": lon_col,
            "zoom": plot_args.get("zoom", 10),
        }
    )

    if chart_type == "calor":
        plot_args["mapbox_style"] = "open-street-map"
        _attach_z_if_possible(plot_args, df, fields, lat_col, lon_col)


def _build_plot_args(
    *,
    chart_type: str,
    df: pd.DataFrame,
    fields: list[str],
    title: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:

    df_plot = df.copy()

    plot_args = {
        "data_frame": df_plot,
        "title": title,
        **(config.get("plotly_params") or {}),
    }
    plot_args.pop("type", None)

    BUILDERS = {
        "histograma": _args_histograma,
        "circular": _args_circular,
        "barras": _args_xy,
        "lineas": _args_xy,
        "dispersion": _args_xy,
        "caja": _args_xy,
        "treemap": _args_treemap,
        "puntos": _args_geo,
        "calor": _args_geo,
    }

    builder = BUILDERS.get(chart_type)
    if not builder:
        raise ValueError(f"No hay builder para '{chart_type}'")

    builder(plot_args, df_plot, fields, chart_type)
    return plot_args, df_plot


def _resolve_fields(df: pd.DataFrame, config: dict[str, Any]) -> list[str] | None:
    fields_orig = config.get("campos_involucrados", [])
    fields = [c for c in fields_orig if c in df.columns]

    if fields_orig and not fields:
        st.warning(f"Campos no existen ({fields_orig}).")
        return None

    if not fields:
        st.warning("No hay campos válidos para la visualización.")
        return None

    return fields


def plot(df: pd.DataFrame, config: dict[str, Any]) -> go.Figure | None:
    chart_type = _normalize_chart_type(config.get("tipo_de_visualizacion", ""))
    plot_func = PLOT_FUNCTIONS.get(chart_type)

    if not plot_func:
        st.warning(f"Tipo visualización no soportado: '{config.get('tipo_de_visualizacion', '')}'")
        return None

    fields = _resolve_fields(df, config)
    if fields is None:
        return None

    title = config.get("titulo_de_la_visualizacion", chart_type)

    try:
        plot_args, df_plot = _build_plot_args(
            chart_type=chart_type,
            df=df,
            fields=fields,
            title=title,
            config=config,
        )
        return plot_func(**plot_args)

    except Exception as e:
        st.error(f"Error generando el gráfico '{title}': {e}")
        return None
