import logging
from typing import Any

import pandas as pd
import streamlit as st
from scipy import io

from biaas.utils import sanitize_filename


def load_dataset_from_bytes(
    dataset_bytes: bytes, dataset_title: str = "dataset"
) -> pd.DataFrame | None:
    file_name_hint = sanitize_filename(dataset_title)

    delimiters = [";", ","]
    encodings = ["utf-8", "latin1"]

    last_error = None

    for encoding in encodings:
        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    io.BytesIO(dataset_bytes),
                    delimiter=delimiter,
                    encoding=encoding,
                    on_bad_lines="warn",
                )

                if df.shape[1] <= 1 and not df.empty:
                    raise ValueError("CSV con una sola columna")

                if df.empty:
                    return None

                df.columns = df.columns.astype(str).str.strip().str.replace(" ", "_").str.lower()

                return df
            except Exception as e:
                last_error = e
                continue
    st.error(f"Error parseando CSV '{file_name_hint}': {last_error}")

    return None


def _preprocess_numric_strings(df: pd.DataFrame) -> None:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False).str.strip())
            except Exception:
                pass


def _preprocess_temporal(df: pd.DataFrame) -> None:
    keywords = ["fecha", "date", "año", "ano", "year", "time"]

    for col in df.columns:
        if not any(k in col.lower() for k in keywords):
            continue

        orig_nn = df[col].notna().sum()
        if orig_nn == 0:
            continue

        try:
            converted = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if converted.notna().sum() / orig_nn > 0.5:
                df[col] = converted
        except Exception:
            pass


def _extract_lat_lon(df: pd.DataFrame, col: str, analysis: dict) -> None:
    if not pd.api.types.is_string_dtype(df[col]):
        return

    try:
        coords = df[col].str.split(",", expand=True)
        if coords.shape[1] < 2:
            return

        df["latitude"] = pd.to_numeric(coords[0], errors="coerce")
        df["longitude"] = pd.to_numeric(coords[1], errors="coerce")

        for c in ("latitude", "longitude"):
            if c not in analysis["numeric"]:
                analysis["numeric"].append(c)
            if c not in analysis["geospatial"]:
                analysis["geospatial"].append(c)

    except Exception as e:
        logging.warning(f"No se pudieron extraer coordenadas de '{col}': {e}")


def _is_geospatial_column(df: pd.DataFrame, col: str, analysis: dict) -> bool:
    cl = col.lower()

    # geopoint packed
    if any(k in cl for k in ("geo_point_2d", "geopoint", "geo_shape")):
        analysis["geospatial"].append(col)
        _extract_lat_lon(df, col, analysis)
        return True

    if any(k in cl for k in ("latitud", "latitude", "longitud", "longitude")):
        analysis["geospatial"].append(col)
        return True

    return False


def analyze_dataset(df: pd.DataFrame) -> dict[str, Any]:
    analysis: dict[str, Any] = {
        "numeric": [],
        "categorical": [],
        "temporal": [],
        "geospatial": [],
        "other": [],
        "stats": None,
        "value_counts": {},
        "temporal_range": {},
    }

    df_copy = df.copy()

    _preprocess_numric_strings(df_copy)
    _preprocess_temporal(df_copy)

    # Análisis de columnas
    for col in df_copy.columns:
        if _is_geospatial_column(df_copy, col, analysis):
            continue

        if pd.api.types.is_numeric_dtype(df_copy[col]):
            analysis["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            analysis["temporal"].append(col)
            analysis["temporal_range"][col] = (df_copy[col].min(), df_copy[col].max())
        elif pd.api.types.is_object_dtype(df_copy[col]) or pd.api.types.is_string_dtype(
            df_copy[col]
        ):
            analysis["categorical"].append(col)
            if 0 < df_copy[col].nunique() < 100:
                try:
                    analysis["value_counts"][col] = df_copy[col].value_counts().to_dict()
                except Exception:
                    pass
        else:
            analysis["other"].append(col)

    try:
        analysis["stats"] = df_copy.describe(include="all").to_dict()
    except Exception as e:
        logging.error(f"Error al generar describe(): {e}")
        analysis["stats"] = {}

    return analysis
