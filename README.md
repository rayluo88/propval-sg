# PropVal SG — Singapore HDB Resale AVM

A production-grade **Automated Valuation Model (AVM)** for Singapore HDB resale flats, with demand forecasting and market trend analysis. Built on 228,000+ real transactions from 2017 to Apr 2026.

**Live demo:** https://propval-sg-775010344611.asia-southeast1.run.app

---

## Overview

Three core capabilities mirror the full lifecycle of a real estate analytics platform:

| Capability | Approach | Performance |
|---|---|---|
| **Property valuation** | LightGBM (production) + sklearn GBR (baseline) | MAPE **6.4%**, R² **0.92** |
| **Market trend analysis** | Monthly aggregation + decomposition by town × flat type | 26 towns, 7 flat types, 2017–2026 |
| **Demand forecasting** | Prophet with SG cooling-measure changepoints | 125 segments, 12-month horizon |

All models are trained on a **time-based holdout** (Jun 2024–Apr 2026 as test set) to simulate real-world deployment conditions, not random splits.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Gradio Dashboard                     │
│   Valuation │ Market Trends │ Price Forecast      │
└──────────────────┬───────────────────────────────┘
                   │ HTTP
┌──────────────────▼───────────────────────────────┐
│                FastAPI Service                    │
│  POST /predict · GET /trends · GET /forecast      │
│  GET /market/overview · GET /meta · GET /health   │
└──────────┬────────────────┬──────────────────────┘
           │                │
    ┌──────▼──────┐  ┌──────▼─────────────────────┐
    │ PostgreSQL  │  │ ML Models                   │
    │ (optional)  │  │ LightGBM AVM · Prophet · GBR│
    └─────────────┘  └─────────────────────────────┘
           ▲
┌──────────┴───────────────────────────────────────┐
│       Data Pipeline — data.gov.sg API            │
│       HDB Resale Flat Prices (2017–present)       │
└──────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Package management | uv |
| AVM — production | LightGBM |
| AVM — baseline | Scikit-learn (GradientBoostingRegressor) |
| Forecasting | Prophet |
| API | FastAPI + Uvicorn |
| Dashboard | Gradio |
| Data store | PostgreSQL (optional) / Parquet (local) |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions |
| Linting | Ruff |
| Testing | pytest |

---

## Project Structure

```
src/
├── config.py            # Centralised settings via pydantic-settings
├── data/
│   ├── ingest.py        # data.gov.sg bulk download → cleaned parquet
│   ├── features.py      # Feature engineering (storey_mid, remaining_lease_years, town_median_psm)
│   └── db.py            # PostgreSQL schema + materialized monthly view
├── avm/
│   ├── train.py         # Time-split training: GBR baseline → LightGBM → eval metrics
│   └── predict.py       # AVMPredictor singleton, PredictionResult dataclass
├── forecast/
│   ├── train.py         # Prophet per-segment with cooling-measure changepoints
│   └── predict.py       # Serves pre-computed forecasts from parquet (no Stan at request time)
├── api/
│   ├── main.py          # FastAPI application with lifespan model loading
│   └── schemas.py       # Pydantic v2 request/response schemas
└── dashboard/
    └── app.py           # Gradio 3-tab interface
```

---

## Quickstart

**Prerequisites:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/rayluo88/propval-sg.git
cd propval-sg
uv sync
```

### 1 — Fetch data

Downloads 228k+ HDB resale transactions from data.gov.sg and saves to `data/hdb_resale.parquet`.

```bash
uv run python -m src.data.ingest
```

### 2 — Train models

```bash
# AVM: sklearn GBR baseline + LightGBM production model → models/
uv run python -m src.avm.train

# Forecasting: Prophet per-segment → models/forecast/ + data/forecasts.parquet
uv run python -m src.forecast.train
```

### 3 — Run the API

```bash
uv run uvicorn src.api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

### 4 — Launch the dashboard

```bash
uv run python -m src.dashboard.app
# Dashboard: http://localhost:7860
```

### Docker (API + Dashboard together)

```bash
docker compose up
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | AVM price prediction with confidence band |
| `GET` | `/trends` | Monthly median price history for a segment |
| `GET` | `/forecast` | 12-month Prophet forecast with CI |
| `GET` | `/market/overview` | Island-wide price + volume trend |
| `GET` | `/meta` | Available towns, flat types, date range |
| `GET` | `/health` | Liveness probe |

**Example — predict resale price:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "town": "ANG MO KIO",
    "flat_type": "4 ROOM",
    "floor_area_sqm": 93,
    "storey_range": "07 TO 09",
    "lease_commence_date": 1985,
    "remaining_lease": "61 years 04 months",
    "flat_model": "IMPROVED"
  }'
```

```json
{
  "predicted_price": 590000,
  "price_low": 543000,
  "price_high": 637000,
  "price_per_sqm": 6344,
  "town": "ANG MO KIO",
  "flat_type": "4 ROOM",
  "floor_area_sqm": 93.0
}
```

---

## Model Details

### AVM — LightGBM (Production)

Trained on 180,800 transactions (Jan 2017–May 2024), evaluated on 47,425 held-out transactions (Jun 2024–Apr 2026).

**Features:**
- `floor_area_sqm` — flat size
- `storey_mid` — midpoint of storey range (e.g. "07 TO 09" → 8.0)
- `remaining_lease_years` — parsed from "61 years 04 months" → 61.33
- `lease_commence_date` — proxy for building age
- `month_year` — decimal year (captures long-run price appreciation)
- `town_median_psm` — town-level median $/sqm, computed on training set only (prevents leakage)
- `town`, `flat_type`, `flat_model`, `storey_band` — categorical

**Test set performance (Jun 2024–Apr 2026):**

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| LightGBM (production) | $42,568 | $56,381 | **6.40%** | **0.9217** |
| GBR baseline (sklearn) | $48,537 | $66,083 | 7.15% | 0.8924 |

### Demand Forecasting — Prophet

- 125 independent models, one per (town, flat_type) segment
- Monthly median resale price as target
- Singapore cooling-measure dates added as explicit changepoints:
  - Jul 2018, Dec 2021, Sep 2022, Apr 2023
- 12-month forward forecast with 80% confidence intervals
- Forecasts pre-computed at training time; served from parquet at inference time (zero Stan overhead per request)

---

## Data Source

**HDB Resale Flat Prices (2017 onwards)**
- Provider: Housing & Development Board via [data.gov.sg](https://data.gov.sg)
- Dataset ID: `d_8b84c4ee58e3cfc0ece0d773c8ca6abc`
- Coverage: Jan 2017–present, updated monthly
- Fields: month, town, flat type, block, street, storey range, floor area, flat model, lease commence date, remaining lease, resale price

---

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

CI runs on every push via GitHub Actions (`.github/workflows/ci.yml`): lint → format check → pytest.

---

## Environment Variables

Copy `.env.example` to `.env` and configure as needed:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/realestate
MODELS_DIR=models
DATA_DIR=data
API_HOST=0.0.0.0
API_PORT=8000
FORECAST_HORIZON_MONTHS=12
```

PostgreSQL is optional — the API and dashboard operate fully from parquet files if `DATABASE_URL` is not set.

---

## License

MIT
