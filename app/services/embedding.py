from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from app.config import Settings, get_settings


class EmbeddingService(Protocol):
    def embed_text(self, text: str) -> list[float]:
        """Return one normalized embedding vector for the input text."""


class HashEmbeddingService:
    """Deterministic offline embedding.

    This is intentionally simple: it lets reviewers run the complete demo
    before downloading BGE. Production configuration should use
    BGEEmbeddingService.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + min(len(token), 20) / 20.0)

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 8) for value in vector]


class BGEEmbeddingService:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in embedding.tolist()]


def build_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    settings = settings or get_settings()
    if settings.embedding_mode == "bge":
        return BGEEmbeddingService(settings.bge_model_name)
    return HashEmbeddingService(settings.embedding_dimension)


def _tokenize(text: str) -> list[str]:
    normalized = text.lower()
    ascii_words = re.findall(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", normalized)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    return ascii_words + chinese_terms

