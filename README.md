# Smart Product Grouper

**Phase 1 — Foundation (vertical slice):** Excel → Clusters → Canonical Labels → Report.

Product data is ingested from Excel (.xlsx) or CSV (.csv), normalized, clustered, assigned canonical labels, and summarized in a report.

## Input data (primary: xlsx, CSV also supported)

- **File:** `data/online_retail_II.xlsx` — place this file in `data/` for the default run.
- **Sheets:** `Year 2009-2010`, `Year 2010-2011` (both are read and combined).
- **Columns:** Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country.
- **CSV support:** `python run.py path/to/file.csv` and `make run INPUT=path/to/file.csv` are supported.

## Setup

**One command for new clones:** `make setup` — creates the conda env `datamining` (if missing) and installs dependencies.

Or step by step:
1. `conda create -n datamining python=3 -y` (if not already created)
2. `conda activate datamining`
3. `make install` or `pip install -r requirements.txt`

## Make commands

| Command | Purpose |
|---------|---------|
| `make setup` | Create conda env + install deps (run once per clone) |
| `make run` | Run pipeline on default xlsx (`data/online_retail_II.xlsx`). Override: `make run INPUT=path/to/file.{xlsx,csv}` |
| `make test` | Run tests (`pytest tests/ -v`) |
| `make demo` | Run pipeline on `data/demo.csv` |
| `make lint` | Lint `src/` with ruff |

All targets use the conda env `datamining` via `conda run`. Or activate `datamining` and run `python run.py data/online_retail_II.xlsx` (or any `.csv`), `pytest tests/ -v`, `ruff check src/` directly.

## Project layout

| Path | Role |
|------|------|
| `src/ingest.py` | Load product data from Excel (.xlsx) |
| `src/normalize.py` | Clean and standardize raw records |
| `src/extract.py` | Extract features for clustering |
| `src/cluster.py` | Group products into clusters |
| `src/canonicalize.py` | Produce canonical labels per cluster |
| `src/evaluate.py` | Evaluate and build report |
| `tests/` | Tests |
| `data/` | Put `online_retail_II.xlsx` here |

Pipeline: **ingest** → **normalize** → **extract** → **cluster** → **canonicalize** → **evaluate** (report).
