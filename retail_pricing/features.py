import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES


def _clean_anomalies(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.sort_values(["item_id", "date"]).reset_index(drop=True).copy()
    replacement = (
        out["local_median_28"]
        .fillna(out["local_mean_28"])
        .fillna(out["sales"])
    )
    out["sales_clean"] = out["sales"].where(
        ~out["is_candidate_anomaly"], replacement
    )
    return out


def _add_lag_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    clean_sales = out.groupby("item_id", sort=False)["sales_clean"]
    for lag in (1, 2, 3, 7, 14, 21, 28):
        out[f"lag_{lag}_clean"] = clean_sales.shift(lag)

    out["roll_mean_7_clean"] = clean_sales.transform(
        lambda values: values.shift(1).rolling(7, min_periods=1).mean()
    )
    out["roll_mean_30_clean"] = clean_sales.transform(
        lambda values: values.shift(1).rolling(30, min_periods=7).mean()
    )
    out["roll_std_30_clean"] = clean_sales.transform(
        lambda values: values.shift(1).rolling(30, min_periods=7).std()
    )
    for window in (7, 30):
        min_periods = max(2, window // 4)
        out[f"roll_median_{window}_clean"] = clean_sales.transform(
            lambda values, size=window, minimum=min_periods: values.shift(1)
            .rolling(size, min_periods=minimum)
            .median()
        )
        out[f"ewm_mean_{window}_clean"] = clean_sales.transform(
            lambda values, span=window, minimum=min_periods: values.shift(1)
            .ewm(span=span, adjust=False, min_periods=minimum)
            .mean()
        )
    out["same_weekday_mean_4"] = out[
        ["lag_7_clean", "lag_14_clean", "lag_21_clean", "lag_28_clean"]
    ].mean(axis=1)
    out["short_trend"] = out["roll_mean_7_clean"] / (
        out["roll_mean_30_clean"] + 1
    )
    out["lag_week_ratio"] = out["lag_1_clean"] / (out["lag_7_clean"] + 1)
    return out


def build_model_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe demand-model features."""
    out = _add_lag_features(_clean_anomalies(scored))
    out["price_cost_ratio"] = out["price"] / out["cost"].clip(lower=1e-6)
    out["price_competitor_ratio"] = (
        out["price"] / out["competitor_price"].clip(lower=1e-6)
    )
    out["item_promo_mean"] = out.groupby(
        ["item_id", "promo_active"], observed=True
    )["sales_clean"].transform(
        lambda values: values.shift(1).expanding(min_periods=3).mean()
    )
    regular_price = out.groupby("item_id", sort=False, observed=True)["price"]
    out["regular_price_90"] = regular_price.transform(
        lambda values: values.shift(1).rolling(90, min_periods=14).median()
    )
    out["price_vs_regular_90"] = out["price"] / out["regular_price_90"]
    out["log_ad_spend"] = np.log1p(out["ad_spend"].clip(lower=0))
    out["month"] = out["date"].dt.month.astype(int)
    out["day_of_week"] = out["date"].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    day_of_year = out["date"].dt.dayofyear.astype(float)
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].astype("category")
    return out
