"""Integration tests for the FastAPI AVM service."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_basic():
    payload = {
        "town": "ANG MO KIO",
        "flat_type": "4 ROOM",
        "floor_area_sqm": 93.0,
        "storey_range": "07 TO 09",
        "lease_commence_date": 1985,
        "remaining_lease": "61 years 04 months",
        "flat_model": "IMPROVED",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 200_000 < data["predicted_price"] < 2_000_000
    assert data["price_low"] < data["predicted_price"] < data["price_high"]
    assert data["price_per_sqm"] > 0


def test_predict_invalid_area():
    payload = {
        "town": "TAMPINES",
        "flat_type": "5 ROOM",
        "floor_area_sqm": -5,  # invalid
        "storey_range": "04 TO 06",
        "lease_commence_date": 1990,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422  # Pydantic validation error


def test_trends_valid_segment():
    r = client.get("/trends", params={"town": "TAMPINES", "flat_type": "4 ROOM"})
    assert r.status_code == 200
    data = r.json()
    assert data["town"] == "TAMPINES"
    assert len(data["history"]) > 0
    first = data["history"][0]
    assert "month" in first
    assert first["median_price"] > 0


def test_trends_invalid_segment():
    r = client.get("/trends", params={"town": "NONEXISTENT", "flat_type": "99 ROOM"})
    assert r.status_code == 404


def test_forecast_valid_segment():
    r = client.get("/forecast", params={"town": "BISHAN", "flat_type": "5 ROOM"})
    assert r.status_code in (200, 404)  # 404 ok if insufficient data for that segment
    if r.status_code == 200:
        data = r.json()
        assert data["forecast_horizon_months"] > 0
        assert len(data["forecast"]) > 0
        pt = data["forecast"][0]
        assert pt["lower_bound"] <= pt["predicted_price"] <= pt["upper_bound"]


def test_market_overview():
    r = client.get("/market/overview")
    assert r.status_code == 200
    data = r.json()
    assert len(data["history"]) > 50  # should have many months of data
    months = [p["month"] for p in data["history"]]
    assert "2017-01" in months


def test_meta():
    r = client.get("/meta")
    assert r.status_code == 200
    data = r.json()
    assert len(data["available_towns"]) == 26
    assert "4 ROOM" in data["available_flat_types"]
    assert data["data_from"] < data["data_to"]
