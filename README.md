# Claims Routing & Loss Severity Modeling

Two models trained on insurance claims data and combined into a single business deliverable: for a new claim, a **ranked list of candidate garages** that balances how likely the customer is to actually go there against how much the repair is expected to cost there.

1. **Severity model** - predicts expected repair cost (`damage_cost_nis`) for a claim.
2. **Garage choice model** - predicts the probability an insured picks a given candidate garage out of their vehicle manufacturer's candidate set.
3. **Combined ranking** - merges the two into a per-claim ranked garage list via a tunable business knob (`lam`) trading recommendation likelihood against expected cost.

Data is synthetic, provided as part of a take-home data science assignment (`data/Data Scientist home assignment.docx`).

## Requirements

Dependency management is via [Poetry](https://python-poetry.org/) (`pyproject.toml` + `poetry.lock`). Python **>=3.14** is required.

```
poetry install
```

| Library | Version constraint |
|---|---|
| pandas | >=3.0.5,<4.0.0 |
| numpy | >=2.5.1,<3.0.0 |
| scikit-learn | >=1.9.0,<2.0.0 |
| matplotlib | >=3.11.1,<4.0.0 |
| seaborn | >=0.13.2,<0.14.0 |
| jupyter | >=1.1.1,<2.0.0 |
| ipykernel | >=7.3.0,<8.0.0 |
| lightgbm | >=4.7.0,<5.0.0 |
| optuna | >=4.9.0,<5.0.0 |
| joblib | >=1.5.3,<2.0.0 |
| pytest (dev) | >=9.1.1,<10.0.0 |

Exact versions used to produce the artifacts committed in `models/` are pinned in `models/metadata.json`: pandas 3.0.5, scikit-learn 1.9.0, lightgbm 4.7.0.

## Project structure

```
claims-ds-pipeline/
├── eda.ipynb                    # Step 1 - EDA & cleaning
├── panel_construction.ipynb     # Step 2 - train/test split & training panels
├── modeling.ipynb               # Step 3 - model training, tuning, ranking, artifacts
├── src/
│   ├── features.py              # shared cleaning/feature logic (train + serve)
│   ├── io_utils.py              # load/save helpers (csv, joblib, json)
│   ├── inference.py             # rank_garages() production scoring path
│   └── __init__.py
├── tests/
│   └── test_inference.py        # pytest coverage of the inference contract
├── data/                        # gitignored - raw inputs + pipeline-generated files
├── models/                      # committed - fitted pipelines + metadata
├── pyproject.toml
└── poetry.lock
```

### `data/` (gitignored - not committed; reproduced by running the notebooks)

| File | What it is |
|---|---|
| `claims.csv` | Raw input - 2,000 partial-damage claims, one row per claim |
| `garages.csv` | Raw input - 150 garages |
| `potential_garages_by_manufacturer.csv` | Raw input - candidate garage set per manufacturer (100 of 150 garages each) |
| `data_dictionary.json` | Column-level documentation for the three raw tables above |
| `Data Scientist home assignment.docx` | Original assignment brief |
| `claims_clean.csv` | Output of `eda.ipynb` - cleaned claims (invalid driver ages flagged, missing coordinates imputed) |
| `severity_panel.csv` | Output of `panel_construction.ipynb` - claim-level training panel for the severity model (2,000 rows) |
| `choice_panel.csv` | Output of `panel_construction.ipynb` - claim × candidate-garage training panel for the choice model (200,000 rows) |
| `city_reference.csv` | Output of `panel_construction.ipynb` - train-only locality lookup, used at inference to fill missing coordinates |
| `garage_share_lookup.csv` | Output of `panel_construction.ipynb` - smoothed, leakage-safe garage popularity prior per (manufacturer, garage) |

### `src/` (shared logic - imported by both notebooks and inference, so train and serve can't drift apart)

| File | What it is |
|---|---|
| `features.py` | `haversine_km`, `flag_invalid_driver_age`, `fill_missing_coordinates`, `clean_claims`, `add_interaction_features` - cleaning rules and feature formulas used identically at training and inference time |
| `io_utils.py` | `load_table`/`save_table` (CSV), `save_model`/`load_model` (joblib), `save_metadata`/`load_metadata` (JSON) |
| `inference.py` | `rank_garages(claim, lam, top_k)` - the production scoring path; validates a raw claim, expands it against its candidate garages, scores with both saved pipelines, and returns the top-k ranked list |

### `tests/`

| File | What it is |
|---|---|
| `test_inference.py` | pytest suite for `rank_garages`: full-claim happy path, coordinate-fallback correctness, driver-age outlier handling, hard failures on missing required fields / unknown manufacturer, and probability-distribution sanity checks. Run with `pytest` from the project root. |

### `models/` (committed artifacts)

| File | What it is |
|---|---|
| `severity_pipeline.pkl` | Fitted sklearn `Pipeline` (imputation + LightGBM regressor wrapped in `TransformedTargetRegressor` for the log target) |
| `choice_pipeline.pkl` | Fitted sklearn `Pipeline` (encoding + LightGBM classifier) |
| `metadata.json` | Feature lists per model, tuned hyperparameters, test-set metrics, and the exact package versions used at training time |

## Pipeline flow

### Step 1 - EDA & cleaning (`eda.ipynb`)

- Loads the three raw tables and validates referential integrity (every `chosen_garage_id` is a valid candidate for its manufacturer; every `garage_id` exists in `garages`).
- Parses dates with an explicit day-first format (avoids pandas silently swapping day/month, which would corrupt the later chronological split). Confirms the data spans 2024–2025 with no negative reporting lags.
- Handles two data quality issues called out in the data dictionary:
  - **Missing insured coordinates (~0.5%)** - imputed with the mean lat/lon of the same `insured_city_id`. Safe to do before the train/test split since a city centroid is static reference data, not a distribution statistic.
  - **Invalid driver ages (~1%, e.g. negative, 12, or sentinel values like 999)** - flagged as `NaN` with a `driver_age_was_imputed` indicator, but **not** imputed here. The actual imputation (median) happens inside the model pipeline, fit on train only, to avoid leaking test information.
- Explores the severity target (`damage_cost_nis`, right-skewed → trained in log scale) against categorical and numeric features, and takes an early look at insured-to-garage haversine distance as a likely choice-model signal.
- Saves `data/claims_clean.csv`.

### Step 2 - Panel construction (`panel_construction.ipynb`)

- Splits claims **chronologically 80/20 by `event_date`** (out-of-time test set, matching how the model is actually used in production), with a representation check confirming no category is starved in either split.
- Builds the **severity panel** (2,000 rows, claim-level, including the chosen garage's features so predicted cost can vary by garage).
- Builds the **choice panel** by cross-joining every claim against its manufacturer's 100 candidate garages (200,000 rows), labeling the actually-chosen garage `1` and the rest `0`.
- Adds claim-garage interaction features (`dist_km`, `log_dist_km`, `dist_rank`, `ses_delta`, `city_type_match`) via `src/features.py`.
- Builds a **leakage-safe garage popularity feature**: train-only, leave-one-out (a claim's own choice is excluded from its garage's count), and smoothed toward the uniform prior (`m=20`) - an earlier version without leave-one-out leaked the label and inflated results.
- Builds `city_reference.csv`, the train-only lookup used at inference to fill in coordinates for claims that arrive without them.
- Saves both panels plus the two lookup tables.

### Step 3 - Modeling (`modeling.ipynb`)

- **Baselines**: naive median-by-`damage_type` and Ridge regression for severity; nearest-garage rule and logistic regression for choice.
- **LightGBM pipelines** with light Optuna tuning (25 trials): severity uses plain KFold CV (no temporal drift found in Step 2); choice uses **GroupKFold on `claim_id`** so a claim's 100 rows never split across train/validation.
- Train/CV/test comparisons to check for overfitting, calibration checks (decile plot for severity, reliability curve + per-claim probability normalization for choice), and feature importance.
- **Combined ranking**: for every candidate garage of a claim, `score = p_norm * (1 - lam * cost_scaled)`, where `p_norm` is the choice probability normalized within the claim's candidate set, `cost_scaled` is the expected cost min-max scaled within the same set, and `lam` is an exposed business knob (0 = rank purely by likelihood; higher = favor cheaper garages). The multiplicative form prevents a garage the customer would never visit from being "rescued" by being cheap.
- Saves both pipelines and `metadata.json`, then exercises `src/inference.py::rank_garages` end-to-end.

### Inference

`src/inference.py` exposes `rank_garages(claim, lam=0.2, top_k=5)` - the production scoring path. All model features are required and used as given, except three fields with known real-world data-quality issues: `driver_age` (outliers replaced the same way as training, then imputed by the pipeline's train-fitted median) and `insured_lat`/`insured_lon` (filled from the train-built city centroid if missing; left `NaN` for an unknown city, since the tree models handle missing distance natively). Candidate expansion, distances, and the popularity lookup are all resolved internally using the same `src/features.py` code the training panels used.

```python
from src.inference import rank_garages

claim = {
    "car_manufacturer": "Toyota",
    "vehicle_age": 4,
    "vehicle_value_nis": 95000,
    "driver_age": 34,
    "damage_type": "collision",
    "opened_by": "insured",
    "insured_city_id": 12,
    "insured_city_type": "urban",
    "insured_city_ses_cluster": 7,
}
rank_garages(claim, lam=0.2, top_k=5)
```

## Results

### Severity model (test set)

| Model | MAE (NIS) | RMSE | WAPE | R² |
|---|---|---|---|---|
| Naive (median by `damage_type`) | 2174.2 | 2805.7 | 0.273 | 0.392 |
| Ridge | 2162.2 | 2787.9 | 0.272 | 0.400 |
| **LightGBM (final)** | **2154.9** | **2819.5** | **0.271** | **0.386** |

Train/CV/test MAE (2,307 / 2,438 / 2,155 NIS) sit close together - no overfitting. Tuned params: `n_estimators=200, learning_rate=0.0201, num_leaves=8, min_child_samples=25, subsample=0.698, colsample_bytree=0.865, reg_lambda=1.626`.

### Garage choice model (test set - ranking metrics, 1 positive out of 100 candidates per claim)

| Model | recall@1 | recall@3 | recall@5 | MRR | mean rank |
|---|---|---|---|---|---|
| Nearest-garage rule | 0.458 | 0.755 | 0.868 | 0.631 | 3.032 |
| Logistic regression | 0.465 | 0.755 | 0.868 | 0.634 | 3.045 |
| **LightGBM (final)** | **0.445** | **0.748** | **0.865** | **0.620** | **3.125** |

Log loss: train 0.0231 / CV 0.0256 / test 0.0263. Tuned params: `n_estimators=300, learning_rate=0.0204, num_leaves=36, min_child_samples=20, subsample=0.852, colsample_bytree=0.607, reg_lambda=3.847`.

### Combined ranking - effect of the `lam` knob

| lam | top-3 hit rate | top-1 hit rate | mean cost of top-1 | mean cost of top-3 |
|---|---|---|---|---|
| 0.0 | 0.748 | 0.445 | 7390.08 | 7389.83 |
| 0.2 (default) | 0.750 | 0.445 | 7389.88 | 7389.14 |
| 0.5 | 0.750 | 0.445 | 7389.18 | 7387.95 |
| 1.0 | 0.605 | 0.362 | 7359.58 | 7353.16 |

## Conclusions

- **Severity is driven mostly by `damage_type` and `vehicle_value_nis`.** The tuned GBM only marginally edges the naive median and Ridge baselines - most of the predictable signal is captured by those two fields, and the rest behaves like noise. That's a legitimate finding, not a failure: it means a simple, explainable model is a defensible production option here.
- **Garage choice is dominated by proximity.** The tuned model matches but doesn't beat the nearest-garage rule - there's little signal beyond distance in this data. Its value isn't beating the rule; it's producing a **calibrated probability for every candidate** (which a hard rule can't), which is exactly what the combined ranking needs, plus a framework ready to absorb richer signals (popularity, capacity, service quality) if they become available.
- **The most instructive result was a negative one.** An early version of the choice model appeared to beat the nearest-garage baseline easily - this turned out to be target leakage in the popularity feature, fixed with leave-one-out encoding. Beating a strong simple baseline too easily is a red flag, not a win.
- **The combined ranking makes the cost/likelihood trade-off explicit** through the single `lam` knob, so choosing an operating point is a business decision backed by a table, not something buried in code.
- **Known limitation:** expected cost per candidate is learned only from garages that were actually chosen, so cross-garage cost differences carry selection bias - the cost side of the ranking should be treated as a v1 estimate.

### Next steps (in order of expected value)

1. **Correct selection bias in cross-garage costs** - doubly robust / causal estimation before using cost differences to steer customers at scale.
2. **Replace haversine with real routing distance** - driving time/distance is the actual friction; haversine is a free v1 stand-in.
3. **Gamma/Tweedie objective for severity** - trains directly on the right-skewed NIS target and removes the log-retransformation bias (log1p + expm1 currently produces a conditional median that underestimates expensive claims).
4. **Ranking objective for choice** - LightGBM's `lambdarank` (or a conditional logit) matches the one-of-100 structure more directly than a binary classifier.
5. **Richer garage features** - capacity, ratings, service quality; the data that would let the choice model finally add signal beyond geography.
6. **Monitoring** - the pipelines are ready for deployment as-is; add drift checks on distance and cost distributions to close the loop.

## Running the pipeline

```bash
poetry install
# run the notebooks in order:
#   1. eda.ipynb
#   2. panel_construction.ipynb
#   3. modeling.ipynb
pytest   # from the project root - exercises the inference contract in tests/test_inference.py
```
