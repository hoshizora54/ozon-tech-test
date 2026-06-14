from pathlib import Path

import numpy as np
import pandas as pd

from .config import EXPECTED_COLUMNS


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the retail panel and validate the required source columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    data = pd.read_parquet(path)
    missing = EXPECTED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values(["item_id", "date"]).reset_index(drop=True)


def data_quality_report(data: pd.DataFrame) -> pd.DataFrame:
    """Build compact technical and business data-quality checks."""
    expected_rows = data["item_id"].nunique() * data["date"].nunique()
    actual_lag = data.groupby("item_id", sort=False)["sales"].shift(1)
    supplied_lag = data.get("sales_lag_1", pd.Series(np.nan, index=data.index))
    comparable = actual_lag.notna() & supplied_lag.notna()
    lag_mismatches = int(
        (~np.isclose(supplied_lag[comparable], actual_lag[comparable])).sum()
    )

    checks = {
        "rows": len(data),
        "expected_full_panel_rows": expected_rows,
        "items": data["item_id"].nunique(),
        "dates": data["date"].nunique(),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_item_dates": int(data.duplicated(["item_id", "date"]).sum()),
        "negative_sales": int((data["sales"] < 0).sum()),
        "nonpositive_prices": int((data["price"] <= 0).sum()),
        "price_below_cost": int((data["price"] < data["cost"]).sum()),
        "sales_above_stock": int((data["sales"] > data["stock_level"]).sum()),
        "sales_lag_1_mismatches": lag_mismatches,
    }
    return pd.DataFrame({"check": checks.keys(), "value": checks.values()})

