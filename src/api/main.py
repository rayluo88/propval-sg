"""
Singapore HDB AVM — FastAPI service.

Endpoints:
  POST /predict          — AVM price prediction
  GET  /trends           — Historical price trend for a segment
  GET  /forecast         — 12-month price forecast for a segment
  GET  /market/overview  — Island-wide monthly median price + volume
  GET  /meta             — Available towns, flat types, date range
  GET  /health           — Liveness probe
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ForecastPoint,
    ForecastResponse,
    MarketOverviewResponse,
    MarketPoint,
    MetaResponse,
    PredictRequest,
    PredictResponse,
    TrendPoint,
    TrendsResponse,
)
from src.avm.predict import get_predictor
from src.forecast.predict import (
    get_all_flat_types,
    get_all_towns,
    get_forecast,
    get_historical_trend,
    get_market_overview,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load models at startup so first request isn't slow
    get_predictor()
    get_all_towns()  # triggers loading of aggregation data
    yield


app = FastAPI(
    title="Singapore HDB AVM API",
    description=(
        "Automated Valuation Model for HDB resale flats. "
        "Predicts resale prices, tracks market trends, and forecasts future prices "
        "using LightGBM (AVM) and Prophet (forecasting) trained on data.gov.sg transactions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, summary="AVM price prediction")
def predict(req: PredictRequest):
    """
    Predict the resale price of an HDB flat using the production LightGBM AVM.
    Returns predicted price with ±8% confidence band.
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(
            town=req.town,
            flat_type=req.flat_type,
            floor_area_sqm=req.floor_area_sqm,
            storey_range=req.storey_range,
            lease_commence_date=req.lease_commence_date,
            remaining_lease=req.remaining_lease,
            flat_model=req.flat_model,
            month=req.month,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return PredictResponse(
        predicted_price=result.predicted_price,
        price_low=result.price_low,
        price_high=result.price_high,
        price_per_sqm=result.price_per_sqm,
        town=req.town.upper(),
        flat_type=req.flat_type.upper(),
        floor_area_sqm=req.floor_area_sqm,
    )


@app.get("/trends", response_model=TrendsResponse, summary="Historical price trends")
def trends(
    town: str = Query(..., examples=["ANG MO KIO"]),
    flat_type: str = Query(..., examples=["4 ROOM"]),
):
    """Monthly median price history for a given (town, flat_type) segment (2017–present)."""
    df = get_historical_trend(town, flat_type)
    if df.empty:
        raise HTTPException(404, f"No data for town='{town}' flat_type='{flat_type}'")

    history = [
        TrendPoint(
            month=row.month.strftime("%Y-%m"),
            median_price=float(row.median_price),
            transaction_count=int(row.transaction_count),
            avg_psm=float(row.avg_psm) if row.avg_psm else 0.0,
        )
        for row in df.itertuples()
    ]
    return TrendsResponse(town=town.upper(), flat_type=flat_type.upper(), history=history)


@app.get("/forecast", response_model=ForecastResponse, summary="Price forecast")
def forecast(
    town: str = Query(..., examples=["ANG MO KIO"]),
    flat_type: str = Query(..., examples=["4 ROOM"]),
):
    """12-month forward price forecast with Prophet confidence intervals."""
    fc = get_forecast(town, flat_type)
    if fc.empty:
        raise HTTPException(
            404,
            f"No forecast available for town='{town}' flat_type='{flat_type}'. "
            "Segment may have too few observations."
        )

    forecast_points = [
        ForecastPoint(
            month=row.ds.strftime("%Y-%m"),
            predicted_price=round(float(row.yhat)),
            lower_bound=round(float(row.yhat_lower)),
            upper_bound=round(float(row.yhat_upper)),
        )
        for row in fc.itertuples()
    ]
    return ForecastResponse(
        town=town.upper(),
        flat_type=flat_type.upper(),
        forecast_horizon_months=len(forecast_points),
        forecast=forecast_points,
    )


@app.get("/market/overview", response_model=MarketOverviewResponse, summary="Market overview")
def market_overview():
    """Island-wide monthly median price and transaction volume (all towns, all flat types)."""
    df = get_market_overview()
    return MarketOverviewResponse(
        history=[
            MarketPoint(
                month=row.month.strftime("%Y-%m"),
                median_price=float(row.median_price),
                total_transactions=int(row.total_transactions),
            )
            for row in df.itertuples()
        ]
    )


@app.get("/meta", response_model=MetaResponse, summary="Available options")
def meta():
    """Returns available towns, flat types, and the data date range."""
    df = get_market_overview()
    months = df["month"].dt.strftime("%Y-%m").tolist() if not df.empty else []
    return MetaResponse(
        available_towns=get_all_towns(),
        available_flat_types=get_all_flat_types(),
        data_from=min(months) if months else "",
        data_to=max(months) if months else "",
    )
