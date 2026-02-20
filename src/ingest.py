"""Load product data from CSV or Excel into pipeline-ready records."""

import pandas as pd

RETAIL_SHEETS = ("Year 2009-2010", "Year 2010-2011")
RETAIL_COLUMNS = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
)


def ingest(path: str) -> list[dict]:
    """Read .xlsx or .csv, normalize column names, return list of dicts per row."""
    if path.lower().endswith(".xlsx"):
        dfs = pd.read_excel(
            path,
            sheet_name=list(RETAIL_SHEETS),
            engine="openpyxl",
        )
        combined = pd.concat(dfs.values(), ignore_index=True)
    elif path.lower().endswith(".csv"):
        combined = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported input format: {path!r}. Supported extensions are .xlsx and .csv."
        )

    combined.columns = [str(c).strip() for c in combined.columns]
    return combined.to_dict(orient="records")
