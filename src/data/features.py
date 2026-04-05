"""
Feature engineering pipeline for AVM model training and inference.

All transformations are deterministic (no data leakage from test set).
Call build_features() to get the full feature matrix from a raw DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ------------------------------------------------------------------
# Feature definitions
# ------------------------------------------------------------------

NUMERIC_FEATURES = [
    "floor_area_sqm",
    "storey_mid",
    "remaining_lease_years",
    "lease_commence_date",
    "month_year",       # decimal year, e.g. 2023.5
    "town_median_psm",  # target-encoded: town-level median price per sqm (train only)
]

CATEGORICAL_FEATURES = [
    "town",
    "flat_type",
    "flat_model",
    "storey_band",      # bucketed storey range
]

TARGET = "resale_price"


def build_features(df: pd.DataFrame, town_psm_map: dict | None = None) -> pd.DataFrame:
    """
    Transform raw cleaned DataFrame into model-ready feature matrix.

    Args:
        df: Cleaned DataFrame from ingest.clean()
        town_psm_map: Dict[town -> median_psm] computed on training set only.
                      Pass None during training (will be computed here and returned).
    Returns:
        Feature DataFrame with all engineered columns.
    """
    df = df.copy()

    # Temporal features
    df["month_year"] = df["month"].dt.year + (df["month"].dt.month - 1) / 12

    # Storey band
    df["storey_band"] = pd.cut(
        df["storey_mid"],
        bins=[0, 6, 12, 18, 25, 35, 50, 100],
        labels=["1-6", "7-12", "13-18", "19-25", "26-35", "36-50", "50+"],
        right=True,
    ).astype(str)

    # Town-level median PSM (target encoding proxy — use training values only)
    if town_psm_map is None:
        psm = df["resale_price"] / df["floor_area_sqm"].replace(0, np.nan)
        town_psm_map = psm.groupby(df["town"]).median().to_dict()

    df["town_median_psm"] = df["town"].map(town_psm_map).fillna(
        df["resale_price"].mean() / df["floor_area_sqm"].mean()
    )

    return df, town_psm_map


def make_preprocessor() -> ColumnTransformer:
    """
    Return a sklearn ColumnTransformer for use inside a Pipeline.
    Numeric features are scaled; categorical features are ordinal-encoded.
    """
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def load_from_parquet(path: str = "data/hdb_resale.parquet") -> pd.DataFrame:
    """Load the locally cached parquet file."""
    df = pd.read_parquet(path)
    df["month"] = pd.to_datetime(df["month"])
    return df
