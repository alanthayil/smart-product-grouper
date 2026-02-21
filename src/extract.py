"""Extract vector features from normalized records."""

from __future__ import annotations

from src.embedding import EmbeddingProvider, OpenAIEmbeddingProvider


def extract(
    records: list[dict],
    provider: EmbeddingProvider | None = None,
) -> list[dict]:
    """Build feature records with one embedding vector per description."""
    if not records:
        return []

    descriptions = [str(record.get("description", "")) for record in records]
    embedding_provider = provider or OpenAIEmbeddingProvider()
    vectors = embedding_provider.embed(descriptions)

    if len(vectors) != len(records):
        raise ValueError("Number of embeddings must match number of input records.")

    features: list[dict] = []
    for index, (record, vector) in enumerate(zip(records, vectors)):
        stock_code = str(record.get("stock_code", "")).strip()
        feature_record: dict[str, object] = {
            "record_id": str(record.get("record_id") or f"record-{index}"),
            "description_norm": descriptions[index],
            "feature_vector": [float(value) for value in vector],
        }
        if stock_code:
            feature_record["stock_code"] = stock_code
        for field in ("unit_value", "unit_name", "unit_system"):
            if record.get(field) is not None:
                feature_record[field] = record[field]
        features.append(feature_record)
    return features
