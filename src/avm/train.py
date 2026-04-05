"""
AVM training pipeline.

Models trained:
  1. Sklearn baseline — GradientBoostingRegressor (fast, interpretable)
  2. LightGBM production — higher accuracy, used in API

Both are evaluated on a 20% time-based holdout (most recent data) to simulate
real-world deployment where we predict future transactions.

Usage:
    uv run python -m src.avm.train
"""

import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

from src.data.features import build_features, load_from_parquet

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Features used by both sklearn and LightGBM
FEATURE_COLS = [
    "floor_area_sqm",
    "storey_mid",
    "remaining_lease_years",
    "lease_commence_date",
    "month_year",
    "town_median_psm",
    "town",
    "flat_type",
    "flat_model",
    "storey_band",
]
TARGET = "resale_price"


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by time — test set is the most recent `test_frac` of months."""
    cutoff = df["month"].quantile(1 - test_frac)
    train = df[df["month"] < cutoff].copy()
    test = df[df["month"] >= cutoff].copy()
    logger.info(
        "Train: %d rows (%s–%s) | Test: %d rows (%s–%s)",
        len(train), train.month.min().date(), train.month.max().date(),
        len(test), test.month.min().date(), test.month.max().date(),
    )
    return train, test


def metrics(y_true, y_pred, label: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    result = {"mae": round(mae), "rmse": round(rmse), "mape": round(mape, 2), "r2": round(r2, 4)}
    logger.info(
        "[%s] MAE=$%s  RMSE=$%s  MAPE=%.2f%%  R²=%.4f",
        label, f"{mae:,.0f}", f"{rmse:,.0f}", mape, r2,
    )
    return result


def train_sklearn_baseline(X_train, y_train, X_test, y_test) -> tuple:
    """GradientBoostingRegressor as sklearn baseline (honours JD's sklearn mention)."""
    # Ordinal-encode categoricals first
    cat_cols = ["town", "flat_type", "flat_model", "storey_band"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
    X_te[cat_cols] = enc.transform(X_te[cat_cols])

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    m = metrics(y_test, preds, label="GBR-baseline")

    # Save
    joblib.dump(
        {"model": model, "encoder": enc, "cat_cols": cat_cols},
        MODELS_DIR / "gbr_baseline.joblib",
    )
    return model, m


def train_lgbm(X_train, y_train, X_test, y_test) -> tuple:
    """LightGBM production model — handles categoricals natively."""
    cat_cols = ["town", "flat_type", "flat_model", "storey_band"]

    X_tr = X_train.copy()
    X_te = X_test.copy()
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype("category")
        X_te[col] = X_te[col].astype("category")

    dtrain = lgb.Dataset(X_tr, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_te, label=y_test, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression",
        "metric": "mape",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    preds = booster.predict(X_te)
    m = metrics(y_test, preds, label="LightGBM-prod")

    booster.save_model(str(MODELS_DIR / "lgbm_avm.txt"))
    logger.info("LightGBM model saved (best iteration: %d)", booster.best_iteration)
    return booster, m


def run() -> dict:
    logger.info("Loading data…")
    df = load_from_parquet()

    logger.info("Building features…")
    train_raw, test_raw = time_split(df)

    train_feat, town_psm_map = build_features(train_raw)
    test_feat, _ = build_features(test_raw, town_psm_map=town_psm_map)

    X_train = train_feat[FEATURE_COLS]
    y_train = train_feat[TARGET]
    X_test = test_feat[FEATURE_COLS]
    y_test = test_feat[TARGET]

    logger.info("Training sklearn GBR baseline…")
    _, gbr_metrics = train_sklearn_baseline(X_train, y_train, X_test, y_test)

    logger.info("Training LightGBM production model…")
    _, lgbm_metrics = train_lgbm(X_train, y_train, X_test, y_test)

    # Save shared artefacts needed at inference time
    joblib.dump(
        {"town_psm_map": town_psm_map, "feature_cols": FEATURE_COLS},
        MODELS_DIR / "feature_meta.joblib",
    )

    results = {"gbr_baseline": gbr_metrics, "lgbm_prod": lgbm_metrics}
    with open(MODELS_DIR / "eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("All models saved to %s", MODELS_DIR)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results = run()
    print("\n=== Final Evaluation ===")
    for name, m in results.items():
        print(f"{name:20s}  MAE=${m['mae']:>8,.0f}  MAPE={m['mape']:.2f}%  R²={m['r2']:.4f}")
