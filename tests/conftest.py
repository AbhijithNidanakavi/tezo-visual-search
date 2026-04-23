from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.search_engine import SearchEngine


@pytest.fixture()
def sample_project() -> Path:
    root = Path(__file__).resolve().parent.parent / ".test_runtime" / str(uuid4())
    (root / "data" / "images").mkdir(parents=True)
    (root / "data" / "indexes").mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def sample_images(sample_project: Path) -> list[Path]:
    images_dir = sample_project / "data" / "images"
    fixtures = [
        ("red_car_city.jpg", (220, 40, 40)),
        ("blue_ocean_beach.jpg", (60, 120, 210)),
        ("green_forest_park.jpg", (60, 150, 80)),
    ]
    paths = []
    for name, color in fixtures:
        image_path = images_dir / name
        Image.new("RGB", (256, 256), color=color).save(image_path)
        paths.append(image_path)
    return paths


@pytest.fixture()
def settings_for_tests(sample_project: Path) -> Settings:
    test_settings = Settings(project_root=sample_project, embedder_backend="hashing")
    test_settings.ensure_directories()
    test_settings.photo_csv_path.write_text(
        "photo_image_url\nhttps://example.com/a.jpg\nhttps://example.com/b.jpg\nhttps://example.com/c.jpg\n",
        encoding="utf-8",
    )
    return test_settings


@pytest.fixture()
def engine(settings_for_tests: Settings, sample_images: list[Path]) -> SearchEngine:
    search_engine = SearchEngine(settings_for_tests, embedder=HashingEmbedder())
    search_engine.build_index(images_dir=settings_for_tests.images_dir, csv_path=settings_for_tests.photo_csv_path)
    return search_engine
