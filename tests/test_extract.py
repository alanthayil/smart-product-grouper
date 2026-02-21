"""Tests for feature extraction and embedding provider behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.embedding import OpenAIEmbeddingProvider
from src.extract import extract


class _FakeProvider:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._vectors


def test_extract_returns_contract_shape_and_order() -> None:
    records = [
        {"description": "white hanging heart holder", "stock_code": "85123A"},
        {"description": "party bunting"},
    ]
    provider = _FakeProvider([[0.1, 0.2], [0.3, 0.4]])

    features = extract(records, provider=provider)

    assert features == [
        {
            "record_id": "record-0",
            "description_norm": "white hanging heart holder",
            "stock_code": "85123A",
            "feature_vector": [0.1, 0.2],
        },
        {
            "record_id": "record-1",
            "description_norm": "party bunting",
            "feature_vector": [0.3, 0.4],
        },
    ]


def test_extract_empty_input_returns_empty_list() -> None:
    assert extract([], provider=_FakeProvider([])) == []


def test_extract_raises_when_provider_count_mismatch() -> None:
    records = [{"description": "a"}, {"description": "b"}]

    with pytest.raises(ValueError, match="Number of embeddings must match"):
        extract(records, provider=_FakeProvider([[0.1, 0.2]]))


def test_openai_provider_calls_api_and_returns_float_vectors(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeEmbeddings:
        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            observed["model"] = model
            observed["input"] = input
            return SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=[1, 2.5, -3]),
                    SimpleNamespace(embedding=[0, 4, 5.75]),
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            observed["api_key"] = api_key
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr("src.embedding.OpenAI", FakeOpenAI)
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    vectors = provider.embed(["first", "second"])

    assert observed == {
        "api_key": "test-key",
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
    }
    assert vectors == [[1.0, 2.5, -3.0], [0.0, 4.0, 5.75]]
