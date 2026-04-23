from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import Settings


class BaseEmbedder(ABC):
    name = "base"

    @abstractmethod
    def embed_text(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        raise NotImplementedError

    def score_labels(self, image_embeddings: np.ndarray, labels: list[str], top_k: int) -> list[list[str]]:
        label_vectors = self.embed_text(labels)
        similarities = image_embeddings @ label_vectors.T
        label_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        return [[labels[i] for i in row] for row in label_indices]


class HashingEmbedder(BaseEmbedder):
    name = "hashing"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _hash_vector(self, text: str, offset: int = 0, span: int | None = None) -> np.ndarray:
        span = span or self.dimension
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().replace("-", " ").split():
            index = offset + (int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % span)
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def embed_text(self, texts: list[str]) -> np.ndarray:
        semantic_span = self.dimension // 2
        return np.vstack([self._hash_vector(text, offset=semantic_span, span=semantic_span) for text in texts])

    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                rgb = img.convert("RGB").resize((32, 32))
                arr = np.asarray(rgb, dtype=np.float32) / 255.0
            hist = []
            for channel in range(3):
                channel_hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0.0, 1.0))
                hist.extend(channel_hist.tolist())
            color_vector = np.array(hist, dtype=np.float32)
            semantic_span = self.dimension // 2
            name_vector = self._hash_vector(image_path.stem.replace("_", " "), offset=semantic_span, span=semantic_span)
            padded = np.zeros(self.dimension, dtype=np.float32)
            limit = min(len(color_vector), semantic_span)
            padded[:limit] = color_vector[:limit]
            vector = (0.15 * padded) + (2.0 * name_vector)
            norm = np.linalg.norm(vector)
            vectors.append(vector if norm == 0 else vector / norm)
        return np.vstack(vectors)


class OpenClipEmbedder(BaseEmbedder):
    name = "open_clip"

    def __init__(self, settings: Settings) -> None:
        import open_clip
        import torch

        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name,
            pretrained=settings.clip_pretrained,
        )
        self._tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
        self._model.eval()

    def _normalize(self, tensor) -> np.ndarray:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True)
        return tensor.cpu().numpy().astype(np.float32)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        with self._torch.no_grad():
            tokens = self._tokenizer(texts)
            embeddings = self._model.encode_text(tokens)
        return self._normalize(embeddings)

    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                tensors.append(self._preprocess(img.convert("RGB")))
        batch = self._torch.stack(tensors)
        with self._torch.no_grad():
            embeddings = self._model.encode_image(batch)
        return self._normalize(embeddings)


def build_embedder(settings: Settings) -> BaseEmbedder:
    if settings.embedder_backend == "hashing":
        return HashingEmbedder()
    return OpenClipEmbedder(settings)
