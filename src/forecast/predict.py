"""
Forecast and trend query interface. Serves pre-computed forecasts
and computes historical trend decomposition on demand.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_agg: pd.DataFrame | None = None
_forecasts: pd.DataFrame | None = None


def _load() -> None:
    global _agg, _forecasts
    if _agg is None:
        _agg = pd.read_parquet("data/hdb_monthly_agg.parquet")
        _agg["month"] = pd.to_datetime(_agg["month"])
    if _forecasts is None:
        _forecasts = pd.read_parquet("data/forecasts.parquet")
        _forecasts["ds"] = pd.to_datetime(_forecasts["ds"])


def get_historical_trend(town: str, flat_type: str) -> pd.DataFrame:
    """
    Return monthly median price history for a given (town, flat_type).
    Columns: month, median_price, transaction_count, avg_psm
    """
    _load()
    mask = (_agg["town"] == town.upper()) & (_agg["flat_type"] == flat_type.upper())
    result = _agg[mask].sort_values("month").reset_index(drop=True)
    return result


def get_forecast(town: str, flat_type: str) -> pd.DataFrame:
    """
    Return 12-month ahead price forecast with confidence bands.
    Columns: ds, yhat, yhat_lower, yhat_upper (future rows only)
    """
    _load()
    mask = (_forecasts["town"] == town.upper()) & (_forecasts["flat_type"] == flat_type.upper())
    fc = _forecasts[mask].sort_values("ds").reset_index(drop=True)

    # Return only the future forecast portion
    latest_history = get_historical_trend(town, flat_type)["month"].max()
    future = fc[fc["ds"] > latest_history]
    return future


def get_all_towns() -> list[str]:
    _load()
    return sorted(_agg["town"].unique().tolist())


def get_all_flat_types() -> list[str]:
    _load()
    return sorted(_agg["flat_type"].unique().tolist())


def get_market_overview() -> pd.DataFrame:
    """
    Island-wide monthly median price and volume across all towns/flat types.
    Used for the top-level market trend chart.
    """
    _load()
    overview = (
        _agg.groupby("month")
        .agg(
            median_price=("median_price", "median"),
            total_transactions=("transaction_count", "sum"),
        )
        .reset_index()
        .sort_values("month")
    )
    return overview
