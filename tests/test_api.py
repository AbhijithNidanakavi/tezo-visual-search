from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def _swap_engine_state(engine) -> dict:
    from app.main import engine as app_engine

    original = app_engine.__dict__.copy()
    app_engine.embeddings = engine.embeddings
    app_engine.records = engine.records
    app_engine.config = engine.config
    app_engine.settings = engine.settings
    app_engine.embedder = engine.embedder
    app_engine.store = engine.store
    return original


def test_health_endpoint(engine) -> None:
    from app.main import engine as app_engine

    original = _swap_engine_state(engine)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["index_ready"] is True
    app_engine.__dict__.update(original)


def test_search_endpoint(engine) -> None:
    from app.main import engine as app_engine

    original = _swap_engine_state(engine)
    client = TestClient(app)
    response = client.get("/search", params={"query": "red car"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["image_id"] == "red_car_city"
    app_engine.__dict__.update(original)
