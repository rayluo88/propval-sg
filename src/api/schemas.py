"""Pydantic request/response schemas for the AVM API."""

from pydantic import BaseModel, Field

# ── AVM ──────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "town": "ANG MO KIO", "flat_type": "4 ROOM", "floor_area_sqm": 93.0,
        "storey_range": "07 TO 09", "lease_commence_date": 1985,
        "remaining_lease": "61 years 04 months", "flat_model": "IMPROVED",
    }}}
    town: str
    flat_type: str
    floor_area_sqm: float = Field(..., gt=0, le=400)
    storey_range: str
    lease_commence_date: int = Field(..., ge=1960, le=2030)
    remaining_lease: str = ""
    flat_model: str = "IMPROVED"
    month: str | None = None


class PredictResponse(BaseModel):
    predicted_price: float
    price_low: float
    price_high: float
    price_per_sqm: float
    town: str
    flat_type: str
    floor_area_sqm: float


# ── Trends ───────────────────────────────────────────────────────────────────

class TrendPoint(BaseModel):
    month: str
    median_price: float
    transaction_count: int
    avg_psm: float


class TrendsResponse(BaseModel):
    town: str
    flat_type: str
    history: list[TrendPoint]


# ── Forecast ─────────────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    month: str
    predicted_price: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    town: str
    flat_type: str
    forecast_horizon_months: int
    forecast: list[ForecastPoint]


# ── Market Overview ───────────────────────────────────────────────────────────

class MarketPoint(BaseModel):
    month: str
    median_price: float
    total_transactions: int


class MarketOverviewResponse(BaseModel):
    history: list[MarketPoint]


# ── Metadata ─────────────────────────────────────────────────────────────────

class MetaResponse(BaseModel):
    available_towns: list[str]
    available_flat_types: list[str]
    data_from: str
    data_to: str
