# Smart Product Grouper

**Phase 1 — Foundation (vertical slice):** CSV → Clusters → Canonical Labels → Report.

Product data is ingested from CSV, normalized, clustered, assigned canonical labels, and summarized in a report.

## Setup

1. **Create and activate the conda environment**
   - `conda create -n datamining python=3 -y`   (if not already created)
   - `conda activate datamining`

2. **Install dependencies**
   - With Make: `make install`
   - Or: `pip install -r requirements.txt`

3. **Run tests**
   - `make test` or `pytest tests/`

4. **Lint**
   - `make lint` or `ruff check src/`

All `make` targets use the conda env `datamining` via `conda run -n datamining ...`, so you can run them without activating the env. If you prefer, activate `datamining` and run `pytest tests/ -v`, `ruff check src/`, etc. directly.

## Project layout

| Path | Role |
|------|------|
| `src/ingest.py` | Load product data from CSV |
| `src/normalize.py` | Clean and standardize raw records |
| `src/extract.py` | Extract features for clustering |
| `src/cluster.py` | Group products into clusters |
| `src/canonicalize.py` | Produce canonical labels per cluster |
| `src/evaluate.py` | Evaluate and build report |
| `tests/` | Tests |
| `data/` | Put sample CSVs here |

Pipeline: **ingest** → **normalize** → **extract** → **cluster** → **canonicalize** → **evaluate** (report).
