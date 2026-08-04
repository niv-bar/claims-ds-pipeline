"""Tests for the inference path - run from the project root with `pytest`.

They cover the input contract: full input used as-is, the three imputable
fields (driver_age outliers, missing coordinates), and hard failures on
missing required fields.
"""

import numpy as np
import pandas as pd
import pytest

from src.inference import rank_garages


@pytest.fixture(scope="module")
def base_claim() -> dict:
    """A realistic full claim taken from the cleaned data."""
    row = pd.read_csv("data/claims_clean.csv").iloc[0]
    return {
        "car_manufacturer": row["car_manufacturer"],
        "vehicle_age": row["vehicle_age"],
        "vehicle_value_nis": row["vehicle_value_nis"],
        "driver_age": 42,
        "damage_type": row["damage_type"],
        "opened_by": row["opened_by"],
        "insured_city_id": row["insured_city_id"],
        "insured_city_type": row["insured_city_type"],
        "insured_city_ses_cluster": row["insured_city_ses_cluster"],
        "insured_lat": row["insured_lat"],
        "insured_lon": row["insured_lon"],
    }


def test_full_claim_returns_ranked_list(base_claim):
    out = rank_garages(base_claim, lam=0.2, top_k=5)
    assert len(out) == 5
    assert list(out.columns) == [
        "candidate_garage_id", "garage_name", "dist_km", "p_choice", "expected_cost_nis", "score",
    ]
    # sorted by score, no NaN anywhere in the output
    assert out["score"].is_monotonic_decreasing
    assert not out.isnull().any().any()
    assert (out["expected_cost_nis"] > 0).all()


def test_probabilities_are_a_distribution(base_claim):
    out = rank_garages(base_claim, top_k=100)
    assert len(out) == 100
    assert np.isclose(out["p_choice"].sum(), 1.0)
    assert (out["p_choice"] >= 0).all()


def test_missing_coordinates_fall_back_to_city_centroid(base_claim):
    claim = {**base_claim, "insured_lat": None, "insured_lon": None}
    out = rank_garages(claim, top_k=5)
    # in this data every claim carries its city's coordinates, so the centroid
    # fallback must reproduce the full-input result
    full = rank_garages(base_claim, top_k=5)
    pd.testing.assert_frame_equal(out, full)


def test_driver_age_outlier_is_imputed_not_crashing(base_claim):
    claim = {**base_claim, "driver_age": 999}
    out = rank_garages(claim, top_k=5)
    assert len(out) == 5
    assert not out["expected_cost_nis"].isnull().any()


def test_missing_required_field_raises(base_claim):
    claim = {k: v for k, v in base_claim.items() if k != "damage_type"}
    with pytest.raises(ValueError, match="damage_type"):
        rank_garages(claim)


def test_unknown_manufacturer_raises(base_claim):
    claim = {**base_claim, "car_manufacturer": "NotARealBrand"}
    with pytest.raises(ValueError, match="car_manufacturer"):
        rank_garages(claim)


def test_lam_zero_ranks_purely_by_probability(base_claim):
    out = rank_garages(base_claim, lam=0.0, top_k=100)
    assert out["p_choice"].is_monotonic_decreasing
