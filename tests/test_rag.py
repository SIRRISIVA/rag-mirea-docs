import pytest
from fastapi.testclient import TestClient
import numpy as np

import search
from api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["chunks_count"] > 0
    assert "hybrid" in data["methods_supported"]

def test_search_validation_empty_query():
    response = client.post("/api/v1/search", json={"query": "   ", "top_k": 3})
    assert response.status_code == 422 or response.status_code == 400

def test_search_validation_bad_top_k():
    response = client.post("/api/v1/search", json={"query": "стипендия", "top_k": 0})
    assert response.status_code == 422

def test_search_hybrid_returns_valid_response():
    response = client.post(
        "/api/v1/search",
        json={"query": "как оформить академический отпуск", "top_k": 3, "method": "hybrid", "alpha": 0.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "как оформить академический отпуск"
    assert data["method"] == "hybrid"
    assert len(data["results"]) == 3
    assert data["execution_time_ms"] > 0
    first_hit = data["results"][0]
    assert "akadem" in first_hit["source"].lower() or "отпуск" in first_hit["text"].lower()

def test_search_rrf():
    response = client.post(
        "/api/v1/search",
        json={"query": "пересдача экзамена", "top_k": 3, "method": "rrf"}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert data["method"] == "rrf"

def test_minmax_normalization():
    arr = np.array([10.0, 20.0, 30.0])
    norm = search.minmax(arr)
    assert pytest.approx(norm[0]) == 0.0
    assert pytest.approx(norm[1]) == 0.5
    assert pytest.approx(norm[2]) == 1.0

def test_minmax_constant_array():
    arr = np.array([5.0, 5.0, 5.0])
    norm = search.minmax(arr)
    assert np.all(norm == 0.0)

def test_mrr_calculation_bounds():
    # MRR must be in [0, 1] and >= recall@1
    m = search.mrr(search.search_emb, k=10)
    r1 = search.recall_at_k(search.search_emb, k=1)
    assert 0.0 <= m <= 1.0
    assert m >= r1
