"""FastAPI app exposing clustering pipeline endpoints."""

from __future__ import annotations

import os
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
try:
    from openai import RateLimitError
except ImportError:  # pragma: no cover - defensive when optional dependency missing
    RateLimitError = None  # type: ignore[assignment]

from src.canonicalize import canonicalize
from src.cluster import cluster
from src.evaluate import evaluate
from src.extract import extract
from src.ingest import ingest
from src.normalize import normalize

app = FastAPI(title="Smart Product Grouper API", version="0.1.0")


@app.post("/cluster")
async def cluster_from_xlsx(file: UploadFile = File(...)) -> dict:
    """Accept an xlsx upload and return pipeline evaluation JSON."""
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Expected a .xlsx file upload.")

    temp_path: str | None = None
    stage = "upload_read"
    try:
        content = await file.read()
        with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(content)
            temp_path = tmp.name

        try:
            stage = "ingest"
            raw = ingest(temp_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid xlsx file: {exc}") from exc

        try:
            stage = "normalize"
            normalized = normalize(raw)
            stage = "extract"
            features = extract(normalized)
            stage = "cluster"
            clusters = cluster(features)
            stage = "canonicalize"
            labels = canonicalize(clusters)
            stage = "evaluate"
            evaluation = evaluate(clusters, labels)
            return evaluation
        except ValueError as exc:
            if "OPENAI_API_KEY" in str(exc):
                raise HTTPException(
                    status_code=500,
                    detail="Server is missing OPENAI_API_KEY for embeddings.",
                ) from exc
            raise HTTPException(status_code=400, detail=f"Invalid input data: {exc}") from exc
        except HTTPException:
            raise
        except Exception as exc:
            if stage == "extract" and (
                (RateLimitError is not None and isinstance(exc, RateLimitError))
                or "insufficient_quota" in str(exc).lower()
                or type(exc).__name__ == "RateLimitError"
            ):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Embedding provider quota/rate limit reached. "
                        "Check OPENAI_API_KEY billing/quota and retry."
                    ),
                ) from exc
            raise HTTPException(
                status_code=500,
                detail="Unexpected server error while processing upload.",
            ) from exc
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
