"""Loading and saving helpers - tables (csv), model artifacts (joblib) and metadata (json).

Paths are relative to the project root by default, so notebooks and scripts run
from the root just work. Everything the inference bundle needs goes through here.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


def load_table(name: str, data_dir: Path = DATA_DIR, **read_kwargs) -> pd.DataFrame:
    """Read a csv from the data directory. Extra kwargs go to pd.read_csv (e.g. parse_dates)."""
    return pd.read_csv(Path(data_dir) / f"{name}.csv", **read_kwargs)


def save_table(df: pd.DataFrame, name: str, data_dir: Path = DATA_DIR) -> Path:
    """Write a csv to the data directory, no index. Returns the path."""
    path = Path(data_dir) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_model(model, name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Persist a fitted pipeline with joblib. Returns the path."""
    path = Path(models_dir) / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(name: str, models_dir: Path = MODELS_DIR):
    """Load a pipeline saved with save_model."""
    return joblib.load(Path(models_dir) / f"{name}.pkl")


def save_metadata(metadata: dict, models_dir: Path = MODELS_DIR) -> Path:
    path = Path(models_dir) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path


def load_metadata(models_dir: Path = MODELS_DIR) -> dict:
    with open(Path(models_dir) / "metadata.json") as f:
        return json.load(f)
