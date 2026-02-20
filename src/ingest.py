"""Load product data from Excel into pipeline-ready records."""

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
    """Read .xlsx, normalize column names, return list of dicts per row."""
    if path.lower().endswith(".xlsx"):
        dfs = pd.read_excel(
            path,
            sheet_name=list(RETAIL_SHEETS),
            engine="openpyxl",
        )
        combined = pd.concat(dfs.values(), ignore_index=True)
    else:
        raise ValueError(
            f"Unsupported input format: {path!r}. Supported extension is .xlsx."
        )

    combined.columns = [str(c).strip() for c in combined.columns]
    missing_columns = [col for col in RETAIL_COLUMNS if col not in combined.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    combined = combined.dropna(how="all")
    return combined.to_dict(orient="records")
