from __future__ import annotations

import time

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
    app_engine._store_signature = engine._store_signature
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


def test_health_reflects_deleted_index(engine) -> None:
    from app.main import engine as app_engine

    original = _swap_engine_state(engine)
    client = TestClient(app)
    engine.store.embeddings_path.unlink()
    engine.store.metadata_path.unlink()
    engine.store.config_path.unlink()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["index_ready"] is False
    assert payload["indexed_images"] == 0
    assert payload["index_image_count"] is None
    app_engine.__dict__.update(original)


def test_search_uses_rebuilt_index_without_restart(settings_for_tests, sample_images) -> None:
    from app.main import engine as app_engine
    from app.search_engine import SearchEngine

    initial_engine = SearchEngine(settings_for_tests, embedder=app_engine.embedder)
    initial_engine.build_index(images_dir=settings_for_tests.images_dir, csv_path=settings_for_tests.photo_csv_path, limit=2)

    original = _swap_engine_state(initial_engine)
    client = TestClient(app)

    time.sleep(0.01)
    rebuilt_count = initial_engine.build_index(images_dir=settings_for_tests.images_dir, csv_path=settings_for_tests.photo_csv_path)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert rebuilt_count == 3
    assert payload["index_ready"] is True
    assert payload["indexed_images"] == 3
    assert payload["index_image_count"] == 3
    app_engine.__dict__.update(original)
