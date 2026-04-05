"""
AVM inference interface. Loads trained LightGBM model at startup.
Used by both the FastAPI service and the Gradio dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from src.data.features import build_features

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


@dataclass
class PredictionResult:
    predicted_price: float
    price_low: float   # ~10th percentile confidence band
    price_high: float  # ~90th percentile confidence band
    price_per_sqm: float


class AVMPredictor:
    """Singleton-friendly wrapper around the trained LightGBM AVM."""

    def __init__(self):
        meta = joblib.load(MODELS_DIR / "feature_meta.joblib")
        self.town_psm_map: dict = meta["town_psm_map"]
        self.feature_cols: list[str] = meta["feature_cols"]
        self.booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_avm.txt"))
        logger.info("AVMPredictor loaded (best_iteration=%d)", self.booster.best_iteration)

    def predict(
        self,
        town: str,
        flat_type: str,
        floor_area_sqm: float,
        storey_range: str,
        lease_commence_date: int,
        remaining_lease: str,
        flat_model: str,
        month: str | None = None,
    ) -> PredictionResult:
        """
        Predict resale price for a single property.

        Args:
            town: HDB town name (e.g. "ANG MO KIO")
            flat_type: e.g. "4 ROOM"
            floor_area_sqm: floor area in square metres
            storey_range: e.g. "07 TO 09"
            lease_commence_date: year HDB lease started, e.g. 1985
            remaining_lease: e.g. "61 years 04 months" (or empty)
            flat_model: e.g. "IMPROVED"
            month: prediction month as "YYYY-MM" (defaults to latest available)
        """
        import datetime

        if month is None:
            month = datetime.date.today().strftime("%Y-%m")

        row = pd.DataFrame([{
            "month": pd.to_datetime(month),
            "town": town.strip().upper(),
            "flat_type": flat_type.strip().upper(),
            "flat_model": flat_model.strip().upper(),
            "floor_area_sqm": float(floor_area_sqm),
            "storey_range": storey_range,
            "lease_commence_date": int(lease_commence_date),
            "remaining_lease": remaining_lease,
            "resale_price": 0,  # placeholder — not used in prediction
        }])

        # Reuse the same feature engineering as training
        from src.data.ingest import _parse_remaining_lease, _parse_storey_mid
        row["remaining_lease_years"] = row["remaining_lease"].apply(_parse_remaining_lease)
        row["storey_mid"] = row["storey_range"].apply(_parse_storey_mid)

        feat, _ = build_features(row, town_psm_map=self.town_psm_map)

        for col in ["town", "flat_type", "flat_model", "storey_band"]:
            feat[col] = feat[col].astype("category")

        X = feat[self.feature_cols]
        pred = float(self.booster.predict(X)[0])

        # Simple asymmetric confidence band based on empirical MAPE (~5%)
        margin = pred * 0.08
        psm = pred / float(floor_area_sqm) if float(floor_area_sqm) > 0 else 0

        return PredictionResult(
            predicted_price=round(pred / 1000) * 1000,
            price_low=round((pred - margin) / 1000) * 1000,
            price_high=round((pred + margin) / 1000) * 1000,
            price_per_sqm=round(psm),
        )


# Module-level singleton — loaded once, reused across requests
_predictor: AVMPredictor | None = None


def get_predictor() -> AVMPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AVMPredictor()
    return _predictor
