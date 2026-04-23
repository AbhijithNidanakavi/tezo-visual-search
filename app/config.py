from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    images_dir: Path = field(init=False)
    indexes_dir: Path = field(init=False)
    metadata_path: Path = field(init=False)
    embeddings_path: Path = field(init=False)
    config_path: Path = field(init=False)
    photo_csv_path: Path = field(init=False)
    embedder_backend: str = field(default_factory=lambda: os.getenv("VISUAL_SEARCH_EMBEDDER", "open_clip"))
    clip_model_name: str = field(default_factory=lambda: os.getenv("OPEN_CLIP_MODEL", "ViT-B-32"))
    clip_pretrained: str = field(default_factory=lambda: os.getenv("OPEN_CLIP_PRETRAINED", "laion2b_s34b_b79k"))
    default_top_k: int = field(default_factory=lambda: int(os.getenv("VISUAL_SEARCH_TOP_K", "5")))
    label_top_k: int = field(default_factory=lambda: int(os.getenv("VISUAL_SEARCH_LABEL_TOP_K", "6")))

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.images_dir = self.data_dir / "images"
        self.indexes_dir = self.data_dir / "indexes"
        self.metadata_path = self.indexes_dir / "metadata.json"
        self.embeddings_path = self.indexes_dir / "embeddings.npy"
        self.config_path = self.indexes_dir / "index_config.json"
        self.photo_csv_path = self.project_root / "photos_url.csv"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
