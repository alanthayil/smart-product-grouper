# Smart Product Grouper — Phase 1
# Uses conda environment "datamining". make run defaults to xlsx; override with make run INPUT=path/to/file.

.PHONY: setup install run test demo lint

CONDA_ENV := datamining
INPUT ?= data/online_retail_II.xlsx

setup:
	conda create -n $(CONDA_ENV) python=3 -y
	conda run -n $(CONDA_ENV) pip install -r requirements.txt

install:
	conda run -n $(CONDA_ENV) pip install -r requirements.txt

run:
	conda run -n $(CONDA_ENV) python run.py $(INPUT)

test:
	conda run -n $(CONDA_ENV) pytest tests/ -v

demo:
	conda run -n $(CONDA_ENV) python run.py data/online_retail_II.xlsx

lint:
	conda run -n $(CONDA_ENV) ruff check src/
