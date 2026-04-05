"""
Fetch HDB resale flat prices from data.gov.sg and save locally.

Dataset: Resale Flat Prices (2017 onwards) — d_8b84c4ee58e3cfc0ece0d773c8ca6abc
API:     POST https://api-open.data.gov.sg/v1/public/api/datasets/{id}/poll-download
         Returns a signed S3 URL to download the full CSV in one shot.
Coverage: 2017-01 to present (~228k records, updated monthly)
"""

import io
import logging
from datetime import UTC, datetime

import httpx
import pandas as pd
from sqlalchemy import text

from src.data.db import get_engine

logger = logging.getLogger(__name__)

DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"


def fetch_all_records() -> pd.DataFrame:
    """Download full HDB resale CSV via data.gov.sg bulk download API."""
    logger.info("Requesting download URL from data.gov.sg…")
    with httpx.Client(timeout=30) as client:
        r = client.get(POLL_URL)
        r.raise_for_status()
        download_url = r.json()["data"]["url"]

    logger.info("Downloading CSV from S3…")
    with httpx.Client(timeout=180) as client:
        r = client.get(download_url, follow_redirects=True)
        r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content))
    logger.info("Downloaded %d rows, %d columns", len(df), df.shape[1])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean raw CSV records."""
    df = df.copy()

    df["month"] = pd.to_datetime(df["month"])
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")

    df["remaining_lease_years"] = df["remaining_lease"].apply(_parse_remaining_lease)
    df["storey_mid"] = df["storey_range"].apply(_parse_storey_mid)

    for col in ["town", "flat_type", "flat_model", "street_name", "block"]:
        df[col] = df[col].str.strip().str.upper()

    df.dropna(subset=["resale_price", "floor_area_sqm"], inplace=True)
    return df


def _parse_remaining_lease(s: str) -> float:
    """Convert '61 years 04 months' or '62 years' → fractional years."""
    if pd.isna(s) or s == "":
        return float("nan")
    s = str(s).lower().strip()
    years = 0.0
    if "year" in s:
        years += float(s.split("year")[0].strip())
    if "month" in s:
        months_str = "".join(c for c in s.split("year")[-1].split("month")[0] if c.isdigit())
        if months_str:
            years += float(months_str) / 12
    return round(years, 4)


def _parse_storey_mid(s: str) -> float:
    """Convert '10 TO 12' → 11.0"""
    try:
        lo, hi = (float(x) for x in str(s).split("TO"))
        return (lo + hi) / 2
    except Exception:
        return float("nan")


def load_to_db(df: pd.DataFrame) -> None:
    """Write cleaned records to PostgreSQL, replacing existing data."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE hdb_resale"))
    df.to_sql("hdb_resale", engine, if_exists="append", index=False, chunksize=5_000, method="multi")  # noqa: E501
    logger.info("Loaded %d rows into hdb_resale", len(df))


def run(parquet_path: str = "data/hdb_resale.parquet", load_db: bool = False) -> pd.DataFrame:
    """Full pipeline: download → clean → save parquet → optionally load DB."""
    raw = fetch_all_records()
    df = clean(raw)

    df.to_parquet(parquet_path, index=False)
    logger.info("Saved parquet → %s", parquet_path)

    if load_db:
        load_to_db(df)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Ingest started at %s", datetime.now(UTC).isoformat())
    run()
