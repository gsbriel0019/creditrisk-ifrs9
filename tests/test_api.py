"""Tests for FastAPI endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "CreditRisk-IFRS9" in data["service"]


def test_scoring_endpoint_approval():
    payload = {
        "application_id": "TEST-APP-01",
        "loan_type": "retail_unsecured",
        "loan_amount": 10000.0,
        "tenor_months": 36,
        "annual_income": 85000.0,
        "debt_to_income": 0.18,
        "bureau_score": 750,
        "delinquencies_2yrs": 0,
        "credit_lines_count": 6,
        "revolving_utilization": 0.20,
        "loan_to_value": 0.0
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["application_id"] == "TEST-APP-01"
    assert data["credit_score"] >= 600
    assert data["predicted_pd_12m"] < 0.10
    assert data["decision"] == "APPROVED"


def test_scoring_endpoint_rejection():
    payload = {
        "application_id": "TEST-APP-BAD",
        "loan_type": "retail_unsecured",
        "loan_amount": 35000.0,
        "tenor_months": 48,
        "annual_income": 18000.0,
        "debt_to_income": 0.72,
        "bureau_score": 420,
        "delinquencies_2yrs": 3,
        "credit_lines_count": 3,
        "revolving_utilization": 0.95,
        "loan_to_value": 0.0
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["credit_score"] < 580
    assert len(data["top_adverse_reasons"]) > 0


def test_scorecard_table_endpoint():
    response = client.get("/scorecard")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "Feature" in data[0]
    assert "Assigned Points" in data[0]
