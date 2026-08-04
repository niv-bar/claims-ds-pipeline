"""Inference: from a raw claim to a ranked garage list.

Input contract (see rank_garages):
- All model features are required and used as given, except the three fields
  that showed genuine quality issues in the data:
    * driver_age       - outliers/missing are set to NaN and imputed by the
                         severity pipeline's train-fitted median (any model).
    * insured_lat/lon  - may be NaN; completed from the train-built city
                         reference (city centroid). If the city is unknown,
                         they stay NaN and the tree models handle the missing
                         distance natively.
- Candidate-side features (candidate set, garage attributes, distances,
  popularity share) are never caller-provided - they're resolved internally
  from the artifact bundle.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import add_interaction_features, flag_invalid_driver_age
from src.io_utils import DATA_DIR, MODELS_DIR, load_metadata, load_model, load_table

REQUIRED_FIELDS = [
    "car_manufacturer", "vehicle_age", "vehicle_value_nis", "damage_type",
    "opened_by", "insured_city_id", "insured_city_type", "insured_city_ses_cluster",
]
NULLABLE_FIELDS = ["driver_age", "insured_lat", "insured_lon"]

SHARE_PRIOR = 1 / 100

_BUNDLE: dict = {}


def _load_bundle(data_dir: Path, models_dir: Path) -> dict:
    """Load and cache everything inference needs. The model is the smallest part of it."""
    key = (str(data_dir), str(models_dir))
    if _BUNDLE.get("key") != key:
        metadata = load_metadata(models_dir)
        _BUNDLE.update(
            key=key,
            severity_model=load_model("severity_pipeline", models_dir),
            choice_model=load_model("choice_pipeline", models_dir),
            sev_features=metadata["severity"]["features"],
            cho_features=metadata["choice"]["features"],
            garages=load_table("garages", data_dir),
            potential=load_table("potential_garages_by_manufacturer", data_dir),
            share_lookup=load_table("garage_share_lookup", data_dir),
            city_reference=load_table("city_reference", data_dir).set_index("insured_city_id"),
        )
    return _BUNDLE


def _complete_claim(claim: dict, city_reference: pd.DataFrame) -> dict:
    """Validate required fields, sanitize driver_age, complete missing coordinates from train data."""
    missing = [f for f in REQUIRED_FIELDS if claim.get(f) is None or pd.isna(claim.get(f))]
    if missing:
        raise ValueError(f"missing required claim fields: {missing}")

    claim = {**{f: np.nan for f in NULLABLE_FIELDS}, **claim}
    claim = {k: (np.nan if v is None else v) for k, v in claim.items()}

    # coordinates: use as given; if NaN, fall back to the train-built city centroid
    if pd.isna(claim["insured_lat"]) or pd.isna(claim["insured_lon"]):
        city_id = claim["insured_city_id"]
        if city_id in city_reference.index:
            claim["insured_lat"] = city_reference.loc[city_id, "insured_lat"]
            claim["insured_lon"] = city_reference.loc[city_id, "insured_lon"]
        # unknown city: stays NaN, distance features stay NaN, tree models handle it
    return claim


def rank_garages(
    claim: dict,
    lam: float = 0.2,
    top_k: int = 5,
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """Ranked garage list for one claim: choice probability vs expected cost.

    lam is the business knob: 0 ranks purely by likelihood, larger values push
    cheaper garages up the list. Returns the top_k candidates sorted by score.
    """
    bundle = _load_bundle(Path(data_dir), Path(models_dir))
    claim = _complete_claim(claim, bundle["city_reference"])

    candidates = bundle["potential"][bundle["potential"]["car_manufacturer"] == claim["car_manufacturer"]]
    if candidates.empty:
        raise ValueError(f"unknown car_manufacturer: {claim['car_manufacturer']!r}")

    # expand the claim against its candidate set and attach garage attributes
    frame = candidates.rename(columns={"garage_id": "candidate_garage_id"}).copy()
    for field, value in claim.items():
        if field != "car_manufacturer":
            frame[field] = value
    frame = frame.merge(
        bundle["garages"], left_on="candidate_garage_id", right_on="garage_id", how="left"
    )
    frame["claim_id"] = 0  # single claim, needed for the within-claim distance rank

    # driver_age: same outlier rule as training -> NaN -> train-fitted median imputer in the pipeline
    frame = flag_invalid_driver_age(frame)

    # interaction features with the exact same code the training panel used
    frame = add_interaction_features(frame)

    # popularity share from the train lookup; unseen pairs get the uniform prior
    frame = frame.merge(
        bundle["share_lookup"], on=["car_manufacturer", "candidate_garage_id"], how="left"
    )
    frame["garage_manufacturer_share"] = frame["garage_manufacturer_share"].fillna(SHARE_PRIOR)

    # both models score every candidate
    frame["expected_cost_nis"] = bundle["severity_model"].predict(frame[bundle["sev_features"]])
    frame["p_raw"] = bundle["choice_model"].predict_proba(frame[bundle["cho_features"]])[:, 1]
    frame["p_choice"] = frame["p_raw"] / frame["p_raw"].sum()

    # combined score: likelihood discounted by cost, scaled within the candidate set.
    # multiplicative on purpose: a garage the customer would never go to (p near 0)
    # can't be rescued by being cheap.
    cost = frame["expected_cost_nis"]
    cost_range = cost.max() - cost.min()
    frame["cost_scaled"] = (cost - cost.min()) / (cost_range if cost_range > 0 else 1)
    frame["score"] = frame["p_choice"] * (1 - lam * frame["cost_scaled"])

    out_cols = ["candidate_garage_id", "garage_name", "dist_km", "p_choice", "expected_cost_nis", "score"]
    return (
        frame.sort_values("score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)[out_cols]
    )
