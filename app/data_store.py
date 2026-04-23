from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class IndexedRecord:
    image_id: str
    image_path: str
    source_url: str | None
    labels: list[str]
    dominant_colors: list[str]


class IndexStore:
    def __init__(self, embeddings_path: Path, metadata_path: Path, config_path: Path) -> None:
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.config_path = config_path

    def save(self, embeddings: np.ndarray, metadata: list[dict[str, Any]], config: dict[str, Any]) -> None:
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, embeddings.astype(np.float32))
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def load(self) -> tuple[np.ndarray, list[IndexedRecord], dict[str, Any]]:
        embeddings = np.load(self.embeddings_path).astype(np.float32)
        raw_metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        records = [IndexedRecord(**item) for item in raw_metadata]
        return embeddings, records, config

    def exists(self) -> bool:
        return self.embeddings_path.exists() and self.metadata_path.exists() and self.config_path.exists()
