from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from app.config import Settings
from app.data_store import IndexStore, IndexedRecord
from app.embeddings import BaseEmbedder, build_embedder
from app.explainer import build_explanation
from app.schemas import SearchResponse, SearchResult
from app.taxonomy import VISUAL_LABELS


class SearchEngine:
    def __init__(self, settings: Settings, embedder: BaseEmbedder | None = None) -> None:
        self.settings = settings
        self._embedder = embedder
        self.store = IndexStore(settings.embeddings_path, settings.metadata_path, settings.config_path)
        self.embeddings: np.ndarray | None = None
        self.records: list[IndexedRecord] = []
        self.config: dict[str, str | int] = {}
        self._store_signature: tuple[int | None, int | None, int | None] = self._current_store_signature()

        if self.store.exists():
            self.load()

    def load(self) -> None:
        self.embeddings, self.records, self.config = self.store.load()
        self._store_signature = self._current_store_signature()

    def _current_store_signature(self) -> tuple[int | None, int | None, int | None]:
        paths = (self.store.embeddings_path, self.store.metadata_path, self.store.config_path)
        return tuple(path.stat().st_mtime_ns if path.exists() else None for path in paths)

    def refresh_if_stale(self) -> bool:
        current_signature = self._current_store_signature()
        if current_signature == self._store_signature:
            return False

        if self.store.exists():
            self.load()
        else:
            self.embeddings = None
            self.records = []
            self.config = {}
            self._store_signature = current_signature
        return True

    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    @embedder.setter
    def embedder(self, value: BaseEmbedder) -> None:
        self._embedder = value

    @property
    def is_ready(self) -> bool:
        return self.embeddings is not None and len(self.records) > 0

    def _dominant_colors(self, image_path: Path, top_k: int = 3) -> list[str]:
        palette = {
            "red": np.array([200, 60, 60]),
            "orange": np.array([230, 140, 50]),
            "yellow": np.array([220, 200, 70]),
            "green": np.array([80, 160, 90]),
            "blue": np.array([70, 120, 210]),
            "purple": np.array([140, 90, 170]),
            "pink": np.array([220, 140, 180]),
            "brown": np.array([140, 100, 70]),
            "black": np.array([35, 35, 35]),
            "white": np.array([235, 235, 235]),
            "gray": np.array([150, 150, 150]),
            "gold": np.array([210, 180, 80]),
        }
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
        pixels = arr.reshape(-1, 3)
        mean_pixel = pixels.mean(axis=0)
        scored = sorted(
            ((name, float(np.linalg.norm(mean_pixel - rgb))) for name, rgb in palette.items()),
            key=lambda item: item[1],
        )
        return [name for name, _ in scored[:top_k]]

    def build_index(self, images_dir: Path | None = None, csv_path: Path | None = None, limit: int | None = None) -> int:
        self.settings.ensure_directories()
        images_root = (images_dir or self.settings.images_dir).resolve()
        csv_source = csv_path or self.settings.photo_csv_path
        project_root = self.settings.project_root.resolve()
        image_paths = sorted([path.resolve() for path in images_root.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        if limit is not None:
            image_paths = image_paths[:limit]
        if not image_paths:
            raise FileNotFoundError(f"No images found in {images_root}")

        embeddings = self.embedder.embed_images(image_paths)
        label_sets = self.embedder.score_labels(embeddings, VISUAL_LABELS, top_k=self.settings.label_top_k)

        source_urls: list[str | None] = [None] * len(image_paths)
        if csv_source.exists():
            frame = pd.read_csv(csv_source)
            url_values = frame.iloc[: len(image_paths), 0].tolist()
            source_urls[: len(url_values)] = url_values

        metadata = []
        for idx, image_path in enumerate(image_paths):
            metadata.append(
                {
                    "image_id": image_path.stem,
                    "image_path": str(image_path.relative_to(project_root)).replace("\\", "/"),
                    "source_url": source_urls[idx],
                    "labels": label_sets[idx],
                    "dominant_colors": self._dominant_colors(image_path),
                }
            )

        config = {
            "embedder_backend": self.embedder.name,
            "image_count": len(image_paths),
            "label_top_k": self.settings.label_top_k,
        }
        self.store.save(embeddings, metadata, config)
        self.load()
        return len(image_paths)

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if not self.is_ready:
            raise RuntimeError("Index is not built yet.")

        started = time.perf_counter()
        k = top_k or self.settings.default_top_k
        text_embedding = self.embedder.embed_text([query])[0]
        scores = self.embeddings @ text_embedding
        indices = np.argsort(-scores)[:k]
        results = []
        for index in indices:
            record = self.records[int(index)]
            score = float((scores[int(index)] + 1) / 2)
            explanation = build_explanation(query, record.labels, record.dominant_colors, score)
            results.append(
                SearchResult(
                    image_id=record.image_id,
                    image_path=record.image_path,
                    source_url=record.source_url,
                    score=round(score, 4),
                    explanation=explanation,
                    labels=record.labels,
                    dominant_colors=record.dominant_colors,
                )
            )
        took_ms = round((time.perf_counter() - started) * 1000, 2)
        return SearchResponse(query=query, top_k=k, took_ms=took_ms, results=results)
