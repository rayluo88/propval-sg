"""
Market trend analysis and demand forecasting using Prophet.

For each (town, flat_type) segment we fit a Prophet model on monthly
median price and transaction volume. The models are serialised to disk.

Key design decisions:
- Median price (not mean) is more robust to high-end outliers
- sg_holidays built into Prophet covers SG public holidays
- Cooling-measure events (ABSD increases, stress-test rate changes) are added
  as known changepoints to help Prophet reason about structural breaks

Usage:
    uv run python -m src.forecast.train
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/forecast")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_HORIZON = 12  # months ahead

# Singapore cooling-measure dates (structural breaks in the market)
SG_COOLING_CHANGEPOINTS = [
    "2018-07-01",  # Jul 2018: ABSD raised significantly
    "2021-12-01",  # Dec 2021: Tightened ABSD + TDSR
    "2022-09-01",  # Sep 2022: Raised ABSD for foreigners to 30%
    "2023-04-01",  # Apr 2023: Raised ABSD for foreigners to 60%
]


def build_monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly median price and transaction count per segment."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()

    agg = (
        df.groupby(["month", "town", "flat_type"])
        .agg(
            median_price=("resale_price", "median"),
            avg_price=("resale_price", "mean"),
            transaction_count=("resale_price", "count"),
            avg_psm=("resale_price", lambda x: (x / df.loc[x.index, "floor_area_sqm"]).median()),
        )
        .reset_index()
    )
    return agg


def fit_prophet_for_segment(
    series: pd.DataFrame,
    target_col: str = "median_price",
) -> Prophet | None:
    """
    Fit a Prophet model for a single (town, flat_type) price series.

    Returns None if insufficient data (< 24 monthly observations).
    """
    ts = series[["month", target_col]].rename(columns={"month": "ds", target_col: "y"})
    ts = ts.dropna(subset=["y"]).sort_values("ds")

    if len(ts) < 24:
        return None

    # Use cooling-measure dates as explicit changepoints (must be within training range)
    known_cps = [
        cp for cp in SG_COOLING_CHANGEPOINTS
        if ts["ds"].min() < pd.Timestamp(cp) < ts["ds"].max()
    ]
    model = Prophet(
        changepoint_prior_scale=0.15,  # moderate flexibility for cooling-measure breaks
        changepoints=known_cps if known_cps else None,
        seasonality_prior_scale=5.0,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )

    model.fit(ts)
    return model


def fit_all_segments(agg: pd.DataFrame) -> dict:
    """Fit Prophet models for all (town, flat_type) segments. Returns dict of models."""
    models = {}
    segments = agg.groupby(["town", "flat_type"])
    total = len(segments)

    for i, ((town, flat_type), group) in enumerate(segments):
        key = f"{town}::{flat_type}"
        model = fit_prophet_for_segment(group)
        if model is not None:
            models[key] = model
            if (i + 1) % 20 == 0 or (i + 1) == total:
                logger.info("  Fitted %d / %d segments", i + 1, total)

    logger.info("Total fitted models: %d", len(models))
    return models


def generate_forecasts(models: dict, horizon: int = FORECAST_HORIZON) -> dict:
    """Generate forward-looking forecasts for all fitted segments."""
    forecasts = {}
    for key, model in models.items():
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        fc = model.predict(future)
        forecasts[key] = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    return forecasts


def run() -> dict:
    logger.info("Loading cleaned data…")
    df = pd.read_parquet("data/hdb_resale.parquet")
    df["month"] = pd.to_datetime(df["month"])

    logger.info("Building monthly aggregation…")
    agg = build_monthly_agg(df)
    agg.to_parquet("data/hdb_monthly_agg.parquet", index=False)
    logger.info("Saved monthly_agg: %d rows", len(agg))

    logger.info("Fitting Prophet models for all segments…")
    models = fit_all_segments(agg)

    logger.info("Generating %d-month forecasts…", FORECAST_HORIZON)
    forecasts = generate_forecasts(models)

    # Save all models and forecasts
    joblib.dump(models, MODELS_DIR / "prophet_models.joblib")

    # Save forecasts as a single parquet for fast API serving
    rows = []
    for key, fc in forecasts.items():
        town, flat_type = key.split("::")
        fc = fc.copy()
        fc["town"] = town
        fc["flat_type"] = flat_type
        rows.append(fc)
    all_fc = pd.concat(rows, ignore_index=True)
    all_fc.to_parquet("data/forecasts.parquet", index=False)

    summary = {
        "n_segments": len(models),
        "forecast_horizon_months": FORECAST_HORIZON,
        "segments": list(models.keys())[:10],  # sample
    }
    with open(MODELS_DIR / "forecast_meta.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Forecasting complete. Models → %s", MODELS_DIR)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run()
    print(f"\nFitted {result['n_segments']} segment models")
    print(f"Forecast horizon: {result['forecast_horizon_months']} months ahead")
