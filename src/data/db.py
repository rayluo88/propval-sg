"""Database connection and schema initialisation."""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings


def get_engine() -> Engine:
    return create_engine(settings.database_url, pool_pre_ping=True)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS hdb_resale (
    id                   SERIAL PRIMARY KEY,
    month                DATE        NOT NULL,
    town                 TEXT        NOT NULL,
    flat_type            TEXT        NOT NULL,
    block                TEXT,
    street_name          TEXT,
    storey_range         TEXT,
    storey_mid           NUMERIC(5,1),
    floor_area_sqm       NUMERIC(7,2) NOT NULL,
    flat_model           TEXT,
    lease_commence_date  SMALLINT,
    remaining_lease      TEXT,
    remaining_lease_years NUMERIC(6,4),
    resale_price         INTEGER      NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hdb_month    ON hdb_resale (month);
CREATE INDEX IF NOT EXISTS idx_hdb_town     ON hdb_resale (town);
CREATE INDEX IF NOT EXISTS idx_hdb_flattype ON hdb_resale (flat_type);
CREATE INDEX IF NOT EXISTS idx_hdb_town_month ON hdb_resale (town, month);

-- Aggregated monthly view for trend/forecast queries
CREATE MATERIALIZED VIEW IF NOT EXISTS hdb_monthly_agg AS
SELECT
    date_trunc('month', month)::date AS month,
    town,
    flat_type,
    COUNT(*)                         AS transaction_count,
    ROUND(AVG(resale_price))         AS avg_price,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY resale_price) AS median_price,
    ROUND(AVG(floor_area_sqm), 1)    AS avg_floor_area,
    ROUND(AVG(resale_price / NULLIF(floor_area_sqm, 0))) AS avg_psm
FROM hdb_resale
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

CREATE UNIQUE INDEX IF NOT EXISTS idx_monthly_agg
    ON hdb_monthly_agg (month, town, flat_type);
"""


def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(SCHEMA_SQL))
    print("Database schema initialised.")


if __name__ == "__main__":
    init_db()
