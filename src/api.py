"""FastAPI app exposing clustering pipeline endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from html import escape
import os
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
try:
    from openai import RateLimitError
except ImportError:  # pragma: no cover - defensive when optional dependency missing
    RateLimitError = None  # type: ignore[assignment]

from src.canonicalize import canonicalize
from src.cluster import cluster
from src.config import load_runtime_config
from src.evaluate import evaluate
from src.extract import extract
from src.ingest import ingest
from src.normalize import normalize
from src.synonym_suggestions import analyze_unmatched_tokens

INVALID_XLSX_DETAIL = (
    "Invalid xlsx file upload. Please provide a valid .xlsx workbook "
    "with required sheets/columns."
)
REQUIRED_ENV_VARS = ("OPENAI_API_KEY",)


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment flag with a safe default."""
    raw = str(os.getenv(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    """Read an integer environment setting with a safe default."""
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


CLUSTER_ENABLE_BLOCKING = _env_bool("CLUSTER_ENABLE_BLOCKING", True)
CLUSTER_BLOCKING_SMALL_INPUT_CUTOFF = _env_int("CLUSTER_BLOCKING_SMALL_INPUT_CUTOFF", 300)
CLUSTER_BLOCKING_RARE_TOKEN_MAX_FREQUENCY = _env_int(
    "CLUSTER_BLOCKING_RARE_TOKEN_MAX_FREQUENCY",
    5,
)


def _required_config_status() -> dict[str, bool]:
    """Return required runtime config presence without exposing values."""
    return {name: bool(os.getenv(name)) for name in REQUIRED_ENV_VARS}


def _ensure_required_config() -> None:
    """Fail fast when required runtime configuration is missing."""
    required = _required_config_status()
    missing = [name for name, present in required.items() if not present]
    if missing:
        missing_vars = ", ".join(missing)
        raise RuntimeError(
            f"Missing required configuration: {missing_vars}. "
            "Set OPENAI_API_KEY in a project .env file (OPENAI_API_KEY=...) "
            "or export it in your environment before starting the server."
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load .env values and validate required local config."""
    load_dotenv(override=False)
    _ensure_required_config()
    yield


app = FastAPI(title="Smart Product Grouper API", version="0.1.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def upload_form() -> str:
    """Render a minimal browser upload form for .xlsx files."""
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Smart Product Grouper</title>
  </head>
  <body>
    <h1>Upload .xlsx file</h1>
    <form method="post" action="/cluster/view" enctype="multipart/form-data">
      <input type="file" name="file" accept=".xlsx" required />
      <button type="submit">Cluster</button>
    </form>
  </body>
</html>
"""


@app.get("/health/config")
async def health_config() -> dict[str, dict[str, bool] | bool]:
    """Return non-sensitive required config readiness."""
    required = _required_config_status()
    return {"required": required, "ok": all(required.values())}


async def _run_pipeline_from_upload(
    file: UploadFile,
    *,
    edge_debug: bool = False,
    edge_debug_top_k: int = 20,
) -> dict:
    """Run pipeline for an uploaded workbook and return evaluation payload."""
    runtime_config = load_runtime_config()
    cluster_config = runtime_config["cluster"]
    normalize_config = runtime_config["normalize"]
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
            raise HTTPException(
                status_code=400,
                detail=INVALID_XLSX_DETAIL,
            ) from exc

        try:
            stage = "normalize"
            normalized = normalize(
                raw,
                canonicalize_number_words=normalize_config["number_words"],
                canonicalize_token_splits=normalize_config["token_splits"],
                remove_noise_tokens=normalize_config["noise_tokens"],
                extract_color=normalize_config["extract_color"],
                extract_quantity=normalize_config["extract_quantity"],
            )
            stage = "extract"
            features = extract(normalized)
            stage = "cluster"
            edge_debug_report: dict[str, object] = {}
            try:
                clusters = cluster(
                    features,
                    similarity_threshold=float(cluster_config["similarity_threshold"]),
                    enforce_color_conflict=cluster_config["conflicts"]["enforce_color"],
                    enforce_quantity_conflict=cluster_config["conflicts"]["enforce_quantity"],
                    enable_blocking=CLUSTER_ENABLE_BLOCKING,
                    blocking_small_input_cutoff=CLUSTER_BLOCKING_SMALL_INPUT_CUTOFF,
                    rare_token_max_frequency=CLUSTER_BLOCKING_RARE_TOKEN_MAX_FREQUENCY,
                    edge_debug=edge_debug,
                    edge_debug_top_k=edge_debug_top_k,
                    edge_debug_collector=edge_debug_report,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                # Backward-compatible fallback for test doubles or legacy call sites.
                clusters = cluster(features)
            stage = "canonicalize"
            labels = canonicalize(clusters)
            stage = "evaluate"
            evaluation = evaluate(clusters, labels)
            if edge_debug:
                evaluation["edge_debug"] = edge_debug_report
            evaluation["unmatched_tokens"] = analyze_unmatched_tokens(raw)
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


@app.post("/cluster")
async def cluster_from_xlsx(
    file: UploadFile = File(...),
    edge_debug: bool = False,
    edge_debug_top_k: int = 20,
) -> dict:
    """Accept an xlsx upload and return pipeline evaluation JSON."""
    return await _run_pipeline_from_upload(
        file,
        edge_debug=edge_debug,
        edge_debug_top_k=edge_debug_top_k,
    )


@app.post("/cluster/view", response_class=HTMLResponse)
async def cluster_table_view(
    file: UploadFile = File(...),
    edge_debug: bool = False,
    edge_debug_top_k: int = 20,
) -> str:
    """Accept an xlsx upload and render clustered groups as an HTML table."""
    evaluation = await _run_pipeline_from_upload(
        file,
        edge_debug=edge_debug,
        edge_debug_top_k=edge_debug_top_k,
    )
    cluster_sizes = evaluation.get("cluster_sizes", {})
    labels = evaluation.get("labels", {})
    suspect_clusters = evaluation.get("suspect_clusters", [])
    suspect_by_id = {
        str(entry.get("cluster_id")) for entry in suspect_clusters if entry.get("cluster_id") is not None
    }

    sorted_cluster_ids = sorted(cluster_sizes.keys(), key=int)
    rows = []
    for cluster_id in sorted_cluster_ids:
        label = str(labels.get(cluster_id, ""))
        size = cluster_sizes.get(cluster_id, 0)
        suspect = "Yes" if cluster_id in suspect_by_id else "No"
        rows.append(
            "<tr>"
            f"<td>{escape(str(cluster_id))}</td>"
            f"<td>{escape(label)}</td>"
            f"<td>{escape(str(size))}</td>"
            f"<td>{suspect}</td>"
            "</tr>"
        )

    table_rows = "".join(rows) or (
        '<tr><td colspan="4">No clusters found.</td></tr>'
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Cluster Table View</title>
  </head>
  <body>
    <h1>Cluster Groups</h1>
    <table border="1" cellspacing="0" cellpadding="6">
      <thead>
        <tr>
          <th>Cluster ID</th>
          <th>Label</th>
          <th>Size</th>
          <th>Suspect</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
    <p><a href="/">Upload another file</a></p>
  </body>
</html>
"""
