"""Tests for ingestion behavior and validation."""

from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

from src.ingest import RETAIL_COLUMNS, RETAIL_SHEETS, ingest


def _write_two_sheet_workbook(path: str, df_1: pd.DataFrame, df_2: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_1.to_excel(writer, sheet_name=RETAIL_SHEETS[0], index=False)
        df_2.to_excel(writer, sheet_name=RETAIL_SHEETS[1], index=False)


def test_ingest_raises_for_missing_required_column(tmp_path: Path) -> None:
    file_path = tmp_path / "missing_column.xlsx"
    columns_without_country = [c for c in RETAIL_COLUMNS if c != "Country"]
    df = pd.DataFrame([{col: "x" for col in columns_without_country}])
    _write_two_sheet_workbook(str(file_path), df, df)

    with pytest.raises(ValueError, match="Missing required columns: Country"):
        ingest(str(file_path))


def test_ingest_drops_fully_empty_rows(tmp_path: Path) -> None:
    file_path = tmp_path / "empty_rows.xlsx"
    valid_row = {
        "Invoice": "536365",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 6,
        "InvoiceDate": "2010-12-01 08:26:00",
        "Price": 2.55,
        "Customer ID": "17850",
        "Country": "United Kingdom",
    }
    empty_row = {col: None for col in RETAIL_COLUMNS}
    df = pd.DataFrame([valid_row, empty_row])
    _write_two_sheet_workbook(str(file_path), df, df.iloc[0:0])

    records = ingest(str(file_path))

    assert len(records) == 1
    assert str(records[0]["Invoice"]) == "536365"


def test_ingest_rejects_non_xlsx_input() -> None:
    with pytest.raises(ValueError, match=r"Supported extension is \.xlsx"):
        ingest("data/demo.csv")


def test_ingest_trims_column_name_whitespace(tmp_path: Path) -> None:
    file_path = tmp_path / "trimmed_columns.xlsx"
    row = {
        " Invoice ": "536365",
        " StockCode ": "85123A",
        " Description ": "WHITE HANGING HEART T-LIGHT HOLDER",
        " Quantity ": 6,
        " InvoiceDate ": "2010-12-01 08:26:00",
        " Price ": 2.55,
        " Customer ID ": "17850",
        " Country ": "United Kingdom",
    }
    df = pd.DataFrame([row])
    _write_two_sheet_workbook(str(file_path), df, df.iloc[0:0])

    records = ingest(str(file_path))

    assert len(records) == 1
    assert "Invoice" in records[0]
    assert " Invoice " not in records[0]
