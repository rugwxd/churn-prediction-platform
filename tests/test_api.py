"""Tests for the FastAPI serving endpoint."""

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app, server


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictEndpoint:
    def test_predict_without_model_returns_503(self, client):
        server.model = None  # Ensure model is not loaded

        payload = {
            "user_id": "test_001",
            "login_frequency": 15.0,
            "avg_session_duration_min": 20.0,
            "feature_usage_score": 65.0,
            "support_tickets_total": 2,
            "days_since_last_login": 3.0,
            "monthly_active_days": 20.0,
            "pages_per_session": 8.0,
            "plan_tier": "professional",
            "billing_cycle": "monthly",
            "signup_channel": "organic",
            "mrr": 99.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 503

    def test_predict_validates_input(self, client):
        # Missing required fields
        payload = {"user_id": "test_001"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_validates_plan_tier(self, client):
        payload = {
            "user_id": "test_001",
            "login_frequency": 15.0,
            "avg_session_duration_min": 20.0,
            "feature_usage_score": 65.0,
            "support_tickets_total": 2,
            "days_since_last_login": 3.0,
            "monthly_active_days": 20.0,
            "pages_per_session": 8.0,
            "plan_tier": "invalid_tier",
            "billing_cycle": "monthly",
            "signup_channel": "organic",
            "mrr": 99.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_without_model_returns_503(self, client):
        server.model = None

        payload = {
            "users": [{
                "user_id": "test_001",
                "login_frequency": 15.0,
                "avg_session_duration_min": 20.0,
                "feature_usage_score": 65.0,
                "support_tickets_total": 2,
                "days_since_last_login": 3.0,
                "monthly_active_days": 20.0,
                "pages_per_session": 8.0,
                "plan_tier": "professional",
                "billing_cycle": "monthly",
                "signup_channel": "organic",
                "mrr": 99.0,
            }]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 503
