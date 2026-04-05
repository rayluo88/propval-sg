# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Objective

Job application to **Data Scientist at Real Estate Analytics Pte Ltd (REA)**. Goal: maximize chance of shortlisting and receiving an offer. See `JD.md` for the full job description.

### Application Phases

1. **Pre-interview:** Demo project (complete) + CV update + cover letter + submit
2. **Post-interview invitation:** Interview preparation (technical + behavioral)

---

## Demo Project Summary

Singapore HDB Resale **Automated Valuation Model (AVM)** — three components:

1. **AVM** — LightGBM production model (MAPE 6.4%, R² 0.92) + sklearn GBR baseline
2. **Market Trend Analysis** — Monthly aggregated price/volume by town × flat type
3. **Demand Forecasting** — Prophet per-segment models (125 segments) with SG cooling measure changepoints

**Data:** HDB Resale Flat Prices 2017–present from data.gov.sg (228,225 transactions, updated monthly)

---

## Commands

```bash
# Setup
uv sync

# Run data ingestion (fetches from data.gov.sg, saves data/hdb_resale.parquet)
uv run python -m src.data.ingest

# Train AVM models (LightGBM + sklearn baseline → models/)
uv run python -m src.avm.train

# Train forecast models (Prophet → models/forecast/)
uv run python -m src.forecast.train

# Start API server
uv run uvicorn src.api.main:app --reload

# Launch Gradio dashboard
uv run python -m src.dashboard.app

# Tests
uv run pytest
uv run pytest tests/test_api.py -v   # API only

# Lint / format
uv run ruff check .
uv run ruff format .

# Docker (local)
docker compose up
```

---

## Architecture

```
src/
├── config.py           # Pydantic settings — DATABASE_URL, MODELS_DIR, etc.
├── data/
│   ├── ingest.py       # data.gov.sg → parquet (bulk CSV download via signed S3 URL)
│   ├── features.py     # Feature engineering (storey_mid, remaining_lease_years, town_median_psm)
│   └── db.py           # SQLAlchemy engine + PostgreSQL schema + materialized view
├── avm/
│   ├── train.py        # Training pipeline: time-split → GBR baseline → LightGBM → eval metrics
│   └── predict.py      # AVMPredictor singleton, PredictionResult dataclass
├── forecast/
│   ├── train.py        # Prophet per-(town, flat_type) with cooling-measure changepoints
│   └── predict.py      # Loads pre-computed forecasts.parquet + hdb_monthly_agg.parquet
├── api/
│   ├── main.py         # FastAPI: /predict /trends /forecast /market/overview /meta /health
│   └── schemas.py      # Pydantic request/response schemas
└── dashboard/
    └── app.py          # Gradio 3-tab interface (valuation, trends, forecast)

models/
├── lgbm_avm.txt        # LightGBM booster (production AVM)
├── gbr_baseline.joblib # sklearn GBR + OrdinalEncoder
├── feature_meta.joblib # town_psm_map + feature_cols list
├── eval_metrics.json   # MAE/MAPE/R² for both models
└── forecast/
    ├── prophet_models.joblib
    └── forecast_meta.json

data/
├── hdb_resale.parquet      # 228k cleaned transactions (gitignored)
├── hdb_monthly_agg.parquet # Monthly aggregation by town×flat_type
└── forecasts.parquet       # Pre-computed 12-month Prophet forecasts
```

### Key Design Decisions

- **Time-based train/test split** — test set is the 20% most recent months (Jun 2024–Apr 2026) to simulate production deployment
- **`town_median_psm`** — target-encoded proxy feature computed on train set only; passed explicitly at inference time via `feature_meta.joblib`
- **Pre-computed forecasts** — Prophet models are trained offline; forecasts are stored as parquet for fast API response (no Stan sampling at request time)
- **`data.gov.sg` bulk download** — dataset `d_8b84c4ee58e3cfc0ece0d773c8ca6abc` via `api-open.data.gov.sg/v1/public/api/datasets/{id}/poll-download` returns a signed S3 URL

---

## Target Role Summary (from JD.md)

- **Focus:** Automated Valuation Models, demand forecasting, market trend analysis
- **Stack required:** Python (NumPy, Scikit-learn, Pandas), SQL, AWS/GCP/Azure
- **Closing date:** 11 Apr 2026
- **Salary:** $5,000–$8,000/month
