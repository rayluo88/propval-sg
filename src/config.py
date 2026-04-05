"""Centralised configuration via environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://localhost:5432/realestate"
    models_dir: str = "models"
    data_dir: str = "data"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Forecasting
    forecast_horizon_months: int = 12


settings = Settings()
