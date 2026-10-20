"""Turn text into embedding vectors."""

import hashlib
import math
from typing import Protocol

from rag.config import Settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Real embeddings from the OpenAI API."""

    def __init__(self, api_key: str, model: str, dims: int, batch_size: int = 100):
        from openai import OpenAI

        # max_retries: the client retries rate-limit and server errors itself,
        # with exponential backoff.
        self._client = OpenAI(api_key=api_key, max_retries=5)
        self._model = model
        self._dims = dims
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model, input=batch, dimensions=self._dims
            )
            # Sort by index so vectors line up with inputs, whatever order the API uses.
            vectors.extend(
                item.embedding for item in sorted(response.data, key=lambda d: d.index)
            )
        return vectors


class FakeEmbedder:
    """Deterministic pseudo-random vectors, for testing the pipeline offline.

    No API calls and no cost, but the vectors carry no meaning, so vector search
    results are random. Use it to check that ingest and the database work.
    """

    def __init__(self, dims: int):
        self._dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self._dims:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            values.extend((byte - 127.5) / 127.5 for byte in digest)
            counter += 1
        values = values[: self._dims]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "fake":
        return FakeEmbedder(settings.embedding_dims)
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Add it to .env, or set EMBEDDING_PROVIDER=fake."
            )
        return OpenAIEmbedder(
            settings.openai_api_key, settings.embedding_model, settings.embedding_dims
        )
    raise SystemExit(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
