# Smart Product Grouper — Phase 1
# Uses conda environment "datamining". Without make: conda activate datamining, then pip/pytest/ruff as needed.

.PHONY: install test lint

CONDA_ENV := datamining

install:
	conda run -n $(CONDA_ENV) pip install -r requirements.txt

test:
	conda run -n $(CONDA_ENV) pytest tests/ -v

lint:
	conda run -n $(CONDA_ENV) ruff check src/
