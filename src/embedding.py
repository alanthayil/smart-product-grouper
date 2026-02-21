"""Embedding providers used by the feature extraction stage."""

from __future__ import annotations

import math
import os
from typing import Any, Protocol

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised when dependency is missing
    OpenAI = None  # type: ignore[assignment]


class EmbeddingProvider(Protocol):
    """Interface for text embedding backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each input text."""


class OpenAIEmbeddingProvider:
    """OpenAI-based embedding backend for normalized descriptions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        client: Any | None = None,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key and client is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        if client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai package is required. Install dependencies from requirements.txt."
                )
            self._client = OpenAI(api_key=resolved_api_key)
        else:
            self._client = client
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed text inputs and return float vectors in input order."""
        if not texts:
            return []

        response = self._client.embeddings.create(model=self._model, input=texts)
        vectors = [[float(value) for value in item.embedding] for item in response.data]

        if len(vectors) != len(texts):
            raise ValueError(
                "Embedding provider returned a vector count that does not match inputs."
            )
        for vector in vectors:
            if not vector:
                raise ValueError("Embedding provider returned an empty feature vector.")
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Embedding provider returned non-finite values.")
        return vectors
