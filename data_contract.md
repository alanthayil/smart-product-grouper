# Data Contract (Phase 1)

This document defines the data contract for the Phase 1 vertical slice:
`ingest -> normalize -> extract -> cluster -> canonicalize -> evaluate`.

It covers:
- Required input columns
- Output schema by pipeline stage
- Cluster schema and label mapping

## 1) Required Columns

### 1.1 Source columns

The pipeline requires the following columns (matching `src/ingest.py`):
- `Invoice`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `Price`
- `Customer ID`
- `Country`

For `.xlsx` input, both sheets are read and combined:
- `Year 2009-2010`
- `Year 2010-2011`

### 1.2 Column-level contract

| Column | Required | Type (raw) | Validation rule |
|---|---|---|---|
| `Invoice` | Yes | string or number | Must be present and non-null after string cast |
| `StockCode` | Yes | string | Must be present and non-null |
| `Description` | Yes | string | Must be present; may be empty after trim but not null |
| `Quantity` | Yes | integer-like | Must be parseable as integer |
| `InvoiceDate` | Yes | datetime-like | Must be parseable as timestamp |
| `Price` | Yes | numeric | Must be parseable as float |
| `Customer ID` | Yes | string or number | Must be present; may be null only if explicitly allowed by downstream logic |
| `Country` | Yes | string | Must be present and non-null |

### 1.3 Missing/invalid data handling

Current guaranteed behavior:
- Column names are trimmed in `ingest`.
- Rows are returned as dictionaries with source keys.

Planned contract behavior (to enforce in implementation):
- If any required column is missing, raise a validation error before normalization.
- If `Quantity`, `InvoiceDate`, or `Price` cannot be parsed, either:
  - reject row with explicit reason, or
  - route row to a rejected-records stream (if introduced later).

## 2) Output Schema

## 2.1 Stage interface contract

| Stage | Function | Input | Output | Contract status |
|---|---|---|---|---|
| Ingest | `ingest(path)` | file path (`.xlsx`) | `list[dict]` raw records | Current |
| Normalize | `normalize(records)` | `list[dict]` raw records | `list[dict]` normalized records | Planned |
| Extract | `extract(records)` | `list[dict]` normalized records | `list[dict]` feature records | Planned |
| Cluster | `cluster(records_or_features)` | normalized/feature records | `list[dict]` clustered records | Planned |
| Canonicalize | `canonicalize(clusters)` | clustered records | `dict[int, str]` label map | Planned |
| Evaluate | `evaluate(clusters, canonical_labels)` | clustered records + label map | `dict` report | Planned |

### 2.2 Main record shape (normalized)

Target normalized record fields:

| Field | Type | Notes |
|---|---|---|
| `invoice` | string | Normalized from `Invoice` |
| `stock_code` | string | Normalized from `StockCode` |
| `description` | string | Cleaned text from `Description` |
| `quantity` | int | Parsed from `Quantity` |
| `invoice_ts` | string (ISO-8601) | Parsed from `InvoiceDate` |
| `price` | float | Parsed from `Price` |
| `customer_id` | string | Normalized from `Customer ID` |
| `country` | string | Normalized from `Country` |

Example:

```json
{
  "invoice": "536365",
  "stock_code": "85123A",
  "description": "WHITE HANGING HEART T-LIGHT HOLDER",
  "quantity": 6,
  "invoice_ts": "2010-12-01T08:26:00",
  "price": 2.55,
  "customer_id": "17850",
  "country": "United Kingdom"
}
```

### 2.3 Feature record shape (extract output)

Minimum planned feature schema:

| Field | Type | Notes |
|---|---|---|
| `record_id` | string | Stable ID for join-back |
| `description_norm` | string | Normalized description text |
| `stock_code` | string | Optional categorical signal |
| `feature_vector` | list[float] | Numeric representation for clustering |

Example:

```json
{
  "record_id": "536365|85123A|0",
  "description_norm": "white hanging heart tea light holder",
  "stock_code": "85123A",
  "feature_vector": [0.12, -0.03, 0.55, 0.09]
}
```

### 2.4 Evaluate report shape

Minimum report contract:

| Field | Type | Notes |
|---|---|---|
| `num_records` | int | Total clustered records |
| `num_clusters` | int | Distinct cluster count |
| `cluster_sizes` | dict[str, int] | Cluster ID to size |
| `labels` | dict[str, str] | Cluster ID to canonical label |

Example:

```json
{
  "num_records": 1000,
  "num_clusters": 120,
  "cluster_sizes": {"0": 42, "1": 18},
  "labels": {"0": "white hanging heart candle holder", "1": "party bunting set"}
}
```

## 3) Cluster Schema

### 3.1 Clustered record contract

Each clustered record must include:

| Field | Type | Required | Rule |
|---|---|---|---|
| `record_id` | string | Yes | Must uniquely identify source record within run |
| `cluster_id` | int | Yes | Non-negative integer ID assigned by clustering step |
| `description_norm` | string | Yes | Used for canonical labeling |
| `feature_vector` | list[float] | Yes | Same vector used during clustering |

Example:

```json
{
  "record_id": "536365|85123A|0",
  "cluster_id": 17,
  "description_norm": "white hanging heart tea light holder",
  "feature_vector": [0.12, -0.03, 0.55, 0.09]
}
```

### 3.2 Canonical label mapping contract

`canonicalize` output must be:

- Type: `dict[int, str]`
- Key: `cluster_id`
- Value: `canonical_label` (human-readable, non-empty)

Example:

```json
{
  "17": "white hanging heart tea light holder",
  "18": "vintage party bunting"
}
```

Note: JSON serialization may render integer keys as strings.

### 3.3 Cross-stage consistency rules

Required consistency checks:
- Every `cluster_id` present in clustered records must exist in label mapping.
- Label mapping must not include unknown cluster IDs.
- `evaluate` must compute metrics from the same clustered set and label map.

## 4) Contract Version

- Version: `phase1-v1`
- Scope: vertical-slice baseline for current repository skeleton
- Backward compatibility: best-effort until clustering and labeling implementations are finalized
