"""Shared feature logic - used by the notebooks (training) and by inference.

Keeping this in one place guarantees train/serve consistency: the exact same
cleaning rules and feature formulas run in both worlds.
"""

import numpy as np
import pandas as pd

DRIVER_AGE_MIN = 17
DRIVER_AGE_MAX = 90


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) between two points given lat/lon, vectorized."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def flag_invalid_driver_age(df: pd.DataFrame, min_age: int = DRIVER_AGE_MIN, max_age: int = DRIVER_AGE_MAX) -> pd.DataFrame:
    """Replace implausible driver ages with NaN and flag them. Imputation happens in the model pipeline.

    Values like -5, 0, 12, 120, 180, 999 are input errors / sentinel values rather than real ages.
    The same rule runs at inference, so an outlier age in a new claim gets the same treatment.
    """
    df = df.copy()
    invalid_mask = ~df["driver_age"].between(min_age, max_age)
    df.loc[invalid_mask, "driver_age"] = np.nan
    df["driver_age_was_imputed"] = invalid_mask.astype(int)
    return df


def fill_missing_coordinates(df: pd.DataFrame, lat_col: str, lon_col: str, city_id_col: str) -> pd.DataFrame:
    """Impute missing coordinates using the city's mean lat/lon (static geography), and flag imputed rows."""
    df = df.copy()
    was_missing = df[lat_col].isnull()

    city_lat = df.groupby(city_id_col)[lat_col].transform("mean")
    city_lon = df.groupby(city_id_col)[lon_col].transform("mean")
    df[lat_col] = df[lat_col].fillna(city_lat)
    df[lon_col] = df[lon_col].fillna(city_lon)

    df[f"{lat_col}_was_imputed"] = was_missing.astype(int)
    return df


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the claims table - wraps the functions above."""
    df = flag_invalid_driver_age(df)
    df = fill_missing_coordinates(df, "insured_lat", "insured_lon", "insured_city_id")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Claim-garage interaction features for the choice problem.

    Expects insured_lat/lon, garage_lat/lon, insured/garage city SES and type, and claim_id.
    Used both to build the training panel and to score candidates at inference.
    If coordinates are NaN, the distance features stay NaN (tree models handle this natively).
    """
    df = df.copy()
    df["dist_km"] = haversine_km(
        df["insured_lat"], df["insured_lon"], df["garage_lat"], df["garage_lon"]
    )
    df["log_dist_km"] = np.log1p(df["dist_km"])
    df["ses_delta"] = (df["insured_city_ses_cluster"] - df["garage_city_ses_cluster"]).abs()
    df["city_type_match"] = (df["insured_city_type"] == df["garage_city_type"]).astype(int)
    df["dist_rank"] = df.groupby("claim_id")["dist_km"].rank(method="first")
    return df
