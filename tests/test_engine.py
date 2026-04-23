from __future__ import annotations

import pytest

from app.embeddings import HashingEmbedder
from app.search_engine import SearchEngine


def test_build_index(engine) -> None:
    assert engine.is_ready
    assert len(engine.records) == 3


def test_search_returns_top_results(engine) -> None:
    response = engine.search("blue beach", top_k=2)
    assert response.top_k == 2
    assert len(response.results) == 2
    assert response.results[0].image_id == "blue_ocean_beach"
    assert "blue" in response.results[0].explanation.lower()


def test_empty_query_rejected(engine) -> None:
    with pytest.raises(ValueError):
        engine.search("   ")


def test_missing_index_raises(settings_for_tests) -> None:
    unindexed = SearchEngine(settings_for_tests, embedder=HashingEmbedder())
    unindexed.embeddings = None
    unindexed.records = []
    with pytest.raises(RuntimeError):
        unindexed.search("forest")
