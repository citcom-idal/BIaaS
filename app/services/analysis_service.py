import logging
from typing import Any

import pandas as pd


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
    keywords = {
        "temp": ["fecha", "date", "año", "ano", "year", "time"],
        "geo": ["geo", "lat", "lon", "coord", "wkt", "point", "shape"],
    }
    df_copy = df.copy()
    lat_col, lon_col = "latitude", "longitude"

    # Pre-procesamiento
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            try:
                df_copy[col] = pd.to_numeric(
                    df_copy[col].str.replace(",", ".", regex=False).str.strip()
                )
            except (ValueError, AttributeError):
                pass
        if any(k in col.lower() for k in keywords["temp"]) or pd.api.types.is_datetime64_any_dtype(
            df_copy[col].dtype
        ):
            try:
                orig_nn = df_copy[col].notna().sum()
                if orig_nn == 0:
                    continue
                conv_df = pd.to_datetime(df_copy[col], errors="coerce", dayfirst=True)
                if (conv_df.notna().sum() / orig_nn) > 0.5:
                    df_copy[col] = conv_df
            except Exception:
                pass

    # Análisis de columnas
    for col in df_copy.columns:
        dtype = df_copy[col].dtype
        cl = col.lower()
        is_geo = False

        # <<< LÓGICA GEO REFORZADA >>>
        if any(k in cl for k in ["geo_point_2d", "geopoint", "geo_shape"]):
            analysis["geospatial"].append(col)
            is_geo = True
            try:
                # Asegurarse de que es una columna de strings antes de usar .str
                if pd.api.types.is_string_dtype(df_copy[col]):
                    coords = df_copy[col].str.split(",", expand=True)
                    if (
                        coords.shape[1] >= 2
                    ):  # Comprobar que la división produjo al menos 2 columnas
                        df[lat_col] = pd.to_numeric(coords[0], errors="coerce")
                        df[lon_col] = pd.to_numeric(coords[1], errors="coerce")
                        # Añadir las nuevas columnas al análisis si no están ya
                        if lat_col not in analysis["numeric"]:
                            analysis["numeric"].append(lat_col)
                        if lon_col not in analysis["numeric"]:
                            analysis["numeric"].append(lon_col)
                        if lat_col not in analysis["geospatial"]:
                            analysis["geospatial"].append(lat_col)
                        if lon_col not in analysis["geospatial"]:
                            analysis["geospatial"].append(lon_col)
            except Exception as e:
                logging.warning(f"No se pudieron extraer coordenadas de la columna '{col}': {e}")
                pass

        elif any(k in cl for k in keywords["geo"]):
            if "latitud" in cl or "latitude" in cl:
                analysis["geospatial"].append(col)
                is_geo = True
            if "longitud" in cl or "longitude" in cl:
                analysis["geospatial"].append(col)
                is_geo = True

        if is_geo:
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            analysis["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            analysis["temporal"].append(col)
            analysis["temporal_range"][col] = (df_copy[col].min(), df_copy[col].max())
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
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
